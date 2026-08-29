"""
ENEL Time-Series Forecasting Pipeline
======================================
Loads ENEL data either from a local CSV **or directly from Yahoo Finance**,
engineers features, and compares multiple forecasting models using walk-forward
(time-series) cross-validation.

Targets
-------
- Regression:      next-day Close price
- Classification:  next-day direction (1=Up, 0=Down/Flat)

Usage — Yahoo Finance (recommended, no CSV needed)
---------------------------------------------------
    python enel/enel_forecast.py --ticker ENEL.MI

    # Custom period or exchange suffix
    python enel/enel_forecast.py --ticker ENEL.MI --period 5y
    python enel/enel_forecast.py --ticker ENI.MI  --period 3y

Usage — local CSV file
----------------------
    python enel/enel_forecast.py --data path/to/ENEL.csv

The CSV (if used) is expected to contain columns:
    Date, Open, High, Low, Close, Volume,
    MACD, Signal, MACD_hist, MA100, MA50, MA5, RSI,
    % Change, % Change vs Average

Numeric values may use comma as decimal separator and dot as thousands
separator (Italian locale format).
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("[INFO] yfinance not available — install it with: pip install yfinance")

try:
    import lightgbm as lgb

    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("[INFO] LightGBM not available — using RandomForest as tree-based model.")

try:
    from statsmodels.tsa.arima.model import ARIMA

    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    print("[INFO] statsmodels not available — ARIMA will be skipped.")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_CSV = REPORTS_DIR / "enel_model_comparison.csv"


# ---------------------------------------------------------------------------
# 1. Data loading & parsing
# ---------------------------------------------------------------------------

def _fix_locale_number(s: str) -> str:
    """Convert Italian locale number strings (1.234,56) to float-parseable (1234.56).

    Logic:
    - If the string contains a comma, treat it as Italian locale:
      remove dot thousands-separators, swap decimal comma to dot.
    - Otherwise, strip any trailing/leading whitespace and percent signs only
      (dot is already a decimal separator in standard format).
    """
    s = s.strip().replace("%", "").replace(" ", "")
    if "," in s:
        # Italian locale: dots are thousands separators, comma is decimal
        s = s.replace(".", "").replace(",", ".")
    # Standard format: dot is decimal separator; leave as-is
    return s


def load_data(csv_path: str) -> pd.DataFrame:
    """Load ENEL CSV with locale-aware numeric parsing."""
    raw = pd.read_csv(csv_path, sep=None, engine="python", dtype=str)

    # --- Parse Date ---
    raw["Date"] = pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce")
    raw = raw.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # --- Numeric columns ---
    numeric_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "MACD", "Signal", "MACD_hist", "MA100", "MA50", "MA5", "RSI",
        "% Change", "% Change vs Average",
    ]
    # Keep only columns that exist in the file
    numeric_cols = [c for c in numeric_cols if c in raw.columns]

    for col in numeric_cols:
        raw[col] = raw[col].apply(lambda x: _fix_locale_number(str(x)) if pd.notna(x) else np.nan)
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw = raw.dropna(subset=["Close"]).reset_index(drop=True)
    print(f"[INFO] Loaded {len(raw)} rows from {csv_path}")
    return raw


# ---------------------------------------------------------------------------
# 1b. Yahoo Finance loader
# ---------------------------------------------------------------------------

def _compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute MACD, RSI, and moving-average indicators from OHLCV data."""
    close = df["Close"]

    # --- Moving averages ---
    df["MA5"]   = close.rolling(5).mean()
    df["MA50"]  = close.rolling(50).mean()
    df["MA100"] = close.rolling(100).mean()

    # --- Daily % change ---
    df["% Change"] = close.pct_change() * 100

    # --- MACD (12/26/9 EMA) ---
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"]      = ema12 - ema26
    df["Signal"]    = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["Signal"]

    # --- RSI (14) ---
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # --- % Change vs rolling average (20-day) ---
    df["% Change vs Average"] = (close / close.rolling(20).mean() - 1) * 100

    return df


def load_from_yahoo(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance and compute technical indicators.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol (e.g. 'ENEL.MI' for Borsa Italiana,
        'ENEL' if available on another exchange).
    period : str
        Download period accepted by yfinance: '1y', '2y', '5y', '10y', 'max', etc.
    """
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance is required. Install it with: pip install yfinance")

    print(f"[INFO] Downloading {ticker} from Yahoo Finance (period={period}) ...")
    raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}'. "
            "Check the ticker symbol (e.g. 'ENEL.MI' for Milan exchange) "
            "and your internet connection."
        )

    # Flatten MultiIndex columns produced by yfinance for a single ticker
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.reset_index().rename(columns={"index": "Date", "Adj Close": "Close"})

    # Ensure standard column names
    raw = raw.rename(columns={"Date": "Date"})
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw.sort_values("Date").reset_index(drop=True)

    # Keep only the columns we need (some yfinance versions include extras)
    keep = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in raw.columns]
    raw = raw[keep].copy()

    # Compute technical indicators
    raw = _compute_technical_indicators(raw)
    raw = raw.dropna(subset=["Close"]).reset_index(drop=True)

    print(f"[INFO] Downloaded {len(raw)} rows for {ticker} "
          f"({raw['Date'].min().date()} → {raw['Date'].max().date()})")
    return raw


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag/rolling features; build regression & classification targets."""
    d = df.copy()

    # --- Targets (shift -1 so row t has target for t+1) ---
    d["target_close"] = d["Close"].shift(-1)
    d["target_dir"] = (d["Close"].shift(-1) > d["Close"]).astype(int)

    # --- Daily return ---
    d["return"] = d["Close"].pct_change()

    # --- Lag features ---
    for lag in range(1, 11):
        d[f"close_lag{lag}"] = d["Close"].shift(lag)
        d[f"return_lag{lag}"] = d["return"].shift(lag)

    vol_col = "Volume" if "Volume" in d.columns else None
    if vol_col:
        for lag in range(1, 6):
            d[f"volume_lag{lag}"] = d[vol_col].shift(lag)

    # --- Rolling features ---
    for window in [5, 10, 20]:
        d[f"ret_roll_mean_{window}"] = d["return"].shift(1).rolling(window).mean()
        d[f"ret_roll_std_{window}"] = d["return"].shift(1).rolling(window).std()
        if vol_col:
            d[f"vol_roll_mean_{window}"] = d[vol_col].shift(1).rolling(window).mean()
            d[f"vol_roll_std_{window}"] = d[vol_col].shift(1).rolling(window).std()

    # Keep existing technical indicator columns as features
    tech_cols = [
        "MACD", "Signal", "MACD_hist", "MA100", "MA50", "MA5", "RSI",
        "% Change", "% Change vs Average", "Open", "High", "Low",
    ]
    tech_cols = [c for c in tech_cols if c in d.columns]

    # Drop rows with NaN in targets or core features
    d = d.dropna(subset=["target_close", "target_dir"] + [f"close_lag{i}" for i in range(1, 6)])
    d = d.reset_index(drop=True)
    return d


def get_feature_columns(df: pd.DataFrame) -> list:
    exclude = {"Date", "target_close", "target_dir"}
    return [c for c in df.columns if c not in exclude]


# ---------------------------------------------------------------------------
# 3. Walk-forward evaluation helpers
# ---------------------------------------------------------------------------

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred):
    return float(mean_absolute_percentage_error(y_true, y_pred))


def evaluate_regression_wf(name, model, X, y, n_splits=5):
    """Walk-forward regression evaluation."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes, rmses, mapes = [], [], []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        # Fit scaler on training fold only to avoid leakage
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        maes.append(mean_absolute_error(y_te, pred))
        rmses.append(rmse(y_te, pred))
        mapes.append(mape(y_te, pred))
    return {
        "model": name,
        "task": "regression",
        "MAE": float(np.mean(maes)),
        "RMSE": float(np.mean(rmses)),
        "MAPE": float(np.mean(mapes)),
    }


def evaluate_classification_wf(name, model, X, y, n_splits=5):
    """Walk-forward classification evaluation."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    accs, precs, recs, f1s = [], [], [], []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        # Fit scaler on training fold only to avoid leakage
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        accs.append(accuracy_score(y_te, pred))
        precs.append(precision_score(y_te, pred, zero_division=0))
        recs.append(recall_score(y_te, pred, zero_division=0))
        f1s.append(f1_score(y_te, pred, zero_division=0))
    return {
        "model": name,
        "task": "classification",
        "Accuracy": float(np.mean(accs)),
        "Precision": float(np.mean(precs)),
        "Recall": float(np.mean(recs)),
        "F1": float(np.mean(f1s)),
    }


# ---------------------------------------------------------------------------
# 4. Naive baseline
# ---------------------------------------------------------------------------

def naive_regression_wf(y_close_series, n_splits=5):
    """Naive predictor: Close(t+1) = Close(t)."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    arr = np.array(y_close_series)
    # Build naive pairs: predict row t as row t-1
    maes, rmses, mapes = [], [], []
    for train_idx, test_idx in tscv.split(arr):
        # target is next-day close already; naive = current close = arr shifted
        # arr[test_idx] are next-day closes; naive uses arr[test_idx - 1]
        prev_idx = test_idx - 1
        # Guard against negative index
        valid = prev_idx >= 0
        y_te = arr[test_idx[valid]]
        y_pred = arr[prev_idx[valid]]
        if len(y_te) == 0:
            continue
        maes.append(mean_absolute_error(y_te, y_pred))
        rmses.append(rmse(y_te, y_pred))
        mapes.append(mape(y_te, y_pred))
    return {
        "model": "Naive (Close[t])",
        "task": "regression",
        "MAE": float(np.mean(maes)),
        "RMSE": float(np.mean(rmses)),
        "MAPE": float(np.mean(mapes)),
    }


def naive_classification_wf(df_feat, n_splits=5):
    """Naive direction: predict Up always (majority-class baseline)."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    y = np.array(df_feat["target_dir"])
    accs, precs, recs, f1s = [], [], [], []
    for _, test_idx in tscv.split(y):
        y_te = y[test_idx]
        y_pred = np.ones(len(y_te), dtype=int)  # always predict Up
        accs.append(accuracy_score(y_te, y_pred))
        precs.append(precision_score(y_te, y_pred, zero_division=0))
        recs.append(recall_score(y_te, y_pred, zero_division=0))
        f1s.append(f1_score(y_te, y_pred, zero_division=0))
    return {
        "model": "Naive (Always Up)",
        "task": "classification",
        "Accuracy": float(np.mean(accs)),
        "Precision": float(np.mean(precs)),
        "Recall": float(np.mean(recs)),
        "F1": float(np.mean(f1s)),
    }


# ---------------------------------------------------------------------------
# 5. ARIMA walk-forward (if available)
# ---------------------------------------------------------------------------

def arima_regression_wf(close_series, n_splits=5, order=(2, 1, 2)):
    """ARIMA walk-forward: re-fit on each training window, predict 1 step ahead."""
    if not ARIMA_AVAILABLE:
        return None
    tscv = TimeSeriesSplit(n_splits=n_splits)
    arr = np.array(close_series)
    maes, rmses, mapes = [], [], []
    for train_idx, test_idx in tscv.split(arr):
        train_end = train_idx[-1] + 1  # exclusive end of training window
        preds = []
        truths = []
        for i, t in enumerate(test_idx):
            # Training window: only data up to and including t-1, anchored at fold start
            fit_start = train_idx[0]
            fit_end = train_end + i  # expand by one step at a time (walk-forward)
            train_window = arr[fit_start:fit_end]
            try:
                mdl = ARIMA(train_window, order=order)
                res = mdl.fit()
                fc = float(res.forecast(1)[0])
            except Exception:
                fc = arr[fit_end - 1]  # fallback to last known value
            preds.append(fc)
            truths.append(arr[t])
        maes.append(mean_absolute_error(truths, preds))
        rmses.append(rmse(truths, preds))
        mapes.append(mape(truths, preds))
    return {
        "model": f"ARIMA{order}",
        "task": "regression",
        "MAE": float(np.mean(maes)),
        "RMSE": float(np.mean(rmses)),
        "MAPE": float(np.mean(mapes)),
    }


# ---------------------------------------------------------------------------
# 6. Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(data: "str | pd.DataFrame", n_splits: int = 5, skip_arima: bool = False):
    """Run the full forecasting pipeline.

    Parameters
    ----------
    data : str or pd.DataFrame
        Either a path to a CSV file or a DataFrame already loaded/downloaded.
    """
    # ---- Load data (accept both a path and a pre-loaded DataFrame) ----
    if isinstance(data, pd.DataFrame):
        df_raw = data
    else:
        df_raw = load_data(data)
    # ---- Feature engineering ----
    df = build_features(df_raw)
    feature_cols = get_feature_columns(df)

    X_raw = df[feature_cols].values
    y_reg = df["target_close"].values
    y_clf = df["target_dir"].values

    # Impute remaining NaNs (from rolling windows at boundaries) with column median
    col_medians = np.nanmedian(X_raw, axis=0)
    nan_mask = np.isnan(X_raw)
    X_raw[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    X = X_raw  # Scaling is applied per fold inside evaluation functions

    print(f"\n[INFO] Dataset shape after feature engineering: {df.shape}")
    print(f"[INFO] Number of features: {len(feature_cols)}")
    print(f"[INFO] Walk-forward splits: {n_splits}")
    print(f"[INFO] Target range: Close {y_reg.min():.2f} – {y_reg.max():.2f}\n")

    results = []

    # ---- Regression models ----
    print("=" * 60)
    print("REGRESSION (next-day Close price)")
    print("=" * 60)

    # Naive baseline
    res = naive_regression_wf(df["Close"].values, n_splits=n_splits)
    results.append(res)
    print(f"  {res['model']:35s}  MAE={res['MAE']:.4f}  RMSE={res['RMSE']:.4f}  MAPE={res['MAPE']:.4f}")

    # ARIMA (use raw close series, not scaled)
    if ARIMA_AVAILABLE and not skip_arima and len(df) <= 1000:
        print("  [INFO] Fitting ARIMA (may take a moment)...")
        res = arima_regression_wf(df["Close"].values, n_splits=n_splits)
        if res:
            results.append(res)
            print(f"  {res['model']:35s}  MAE={res['MAE']:.4f}  RMSE={res['RMSE']:.4f}  MAPE={res['MAPE']:.4f}")
    elif ARIMA_AVAILABLE and not skip_arima:
        print("  [INFO] Skipping ARIMA: dataset too large for per-step re-fitting (>1000 rows).")
    elif skip_arima:
        print("  [INFO] ARIMA skipped via --skip-arima flag.")

    # RandomForest Regression
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    res = evaluate_regression_wf("RandomForest Regressor", rf_reg, X, y_reg, n_splits=n_splits)
    results.append(res)
    print(f"  {res['model']:35s}  MAE={res['MAE']:.4f}  RMSE={res['RMSE']:.4f}  MAPE={res['MAPE']:.4f}")

    # LightGBM Regression
    if LGBM_AVAILABLE:
        lgbm_reg = lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            random_state=42, n_jobs=-1, verbose=-1
        )
        res = evaluate_regression_wf("LightGBM Regressor", lgbm_reg, X, y_reg, n_splits=n_splits)
        results.append(res)
        print(f"  {res['model']:35s}  MAE={res['MAE']:.4f}  RMSE={res['RMSE']:.4f}  MAPE={res['MAPE']:.4f}")

    # ---- Classification models ----
    print("\n" + "=" * 60)
    print("CLASSIFICATION (next-day direction: Up=1 / Down=0)")
    print("=" * 60)

    res = naive_classification_wf(df, n_splits=n_splits)
    results.append(res)
    print(f"  {res['model']:35s}  Acc={res['Accuracy']:.4f}  F1={res['F1']:.4f}")

    # Logistic Regression
    lr_clf = LogisticRegression(max_iter=500, random_state=42)
    res = evaluate_classification_wf("LogisticRegression", lr_clf, X, y_clf, n_splits=n_splits)
    results.append(res)
    print(f"  {res['model']:35s}  Acc={res['Accuracy']:.4f}  F1={res['F1']:.4f}")

    # RandomForest Classifier
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    res = evaluate_classification_wf("RandomForest Classifier", rf_clf, X, y_clf, n_splits=n_splits)
    results.append(res)
    print(f"  {res['model']:35s}  Acc={res['Accuracy']:.4f}  F1={res['F1']:.4f}")

    # LightGBM Classifier
    if LGBM_AVAILABLE:
        lgbm_clf = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            random_state=42, n_jobs=-1, verbose=-1
        )
        res = evaluate_classification_wf("LightGBM Classifier", lgbm_clf, X, y_clf, n_splits=n_splits)
        results.append(res)
        print(f"  {res['model']:35s}  Acc={res['Accuracy']:.4f}  F1={res['F1']:.4f}")

    # ---- Save report ----
    report_df = pd.DataFrame(results)
    report_df.to_csv(REPORT_CSV, index=False, float_format="%.4f")
    print(f"\n[INFO] Evaluation report saved to: {REPORT_CSV}")

    # ---- Best model selection ----
    reg_results = [r for r in results if r["task"] == "regression"]
    clf_results = [r for r in results if r["task"] == "classification"]

    best_reg = min(reg_results, key=lambda r: r["MAE"])
    best_clf = max(clf_results, key=lambda r: r["F1"])

    print("\n" + "=" * 60)
    print("BEST MODEL RECOMMENDATION")
    print("=" * 60)
    print(f"\n  📈 REGRESSION  — Best model: {best_reg['model']}")
    print(f"     MAE  = {best_reg['MAE']:.4f}")
    print(f"     RMSE = {best_reg['RMSE']:.4f}")
    print(f"     MAPE = {best_reg['MAPE']:.4f}")
    print(f"\n  🎯 CLASSIFICATION — Best model: {best_clf['model']}")
    print(f"     Accuracy  = {best_clf['Accuracy']:.4f}")
    print(f"     Precision = {best_clf['Precision']:.4f}")
    print(f"     Recall    = {best_clf['Recall']:.4f}")
    print(f"     F1        = {best_clf['F1']:.4f}")
    print("\n" + "=" * 60)

    return report_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="ENEL forecasting pipeline — walk-forward model comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch data directly from Yahoo Finance (ENEL on Borsa Italiana):
  python enel/enel_forecast.py --ticker ENEL.MI

  # Custom period:
  python enel/enel_forecast.py --ticker ENEL.MI --period 10y

  # Use a local CSV file instead:
  python enel/enel_forecast.py --data enel/ENEL.csv

  # Skip slow ARIMA fitting:
  python enel/enel_forecast.py --ticker ENEL.MI --skip-arima
        """,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Yahoo Finance ticker symbol (e.g. ENEL.MI). "
             "When set, data is downloaded automatically — no CSV needed.",
    )
    source.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to a local ENEL CSV dataset. "
             "Ignored when --ticker is provided.",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="5y",
        help="Download period for Yahoo Finance (default: 5y). "
             "Examples: 1y, 2y, 5y, 10y, max. Only used with --ticker.",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=5,
        help="Number of walk-forward splits (default: 5)",
    )
    parser.add_argument(
        "--skip-arima",
        action="store_true",
        help="Skip ARIMA fitting (speeds up the pipeline significantly)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ---- Decide data source ----
    if args.ticker:
        # Yahoo Finance path
        try:
            df_raw = load_from_yahoo(args.ticker, period=args.period)
        except (ValueError, ImportError) as exc:
            print(f"[ERROR] {exc}")
            sys.exit(1)
    else:
        # CSV path (fall back to default location if neither flag given)
        data_path = args.data or "enel/ENEL.csv"
        if not os.path.isabs(data_path):
            candidate = REPO_ROOT / data_path
            if candidate.exists():
                data_path = str(candidate)

        if not os.path.exists(data_path):
            print(f"[ERROR] Dataset not found at: {data_path}")
            print(
                "  Options:\n"
                "    1. Download from Yahoo Finance:  python enel/enel_forecast.py --ticker ENEL.MI\n"
                "    2. Provide a local CSV:          python enel/enel_forecast.py --data /path/to/ENEL.csv"
            )
            sys.exit(1)
        df_raw = load_data(data_path)

    run_pipeline(df_raw, n_splits=args.splits, skip_arima=args.skip_arima)
