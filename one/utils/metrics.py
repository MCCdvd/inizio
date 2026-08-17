import numpy as np
import pandas as pd


def calculate_max_drawdown(portfolio_values):
    values = np.array(portfolio_values, dtype=np.float32)
    if len(values) == 0:
        return 0.0
    peaks = np.maximum.accumulate(values)
    drawdowns = (peaks - values) / np.maximum(peaks, 1e-8)
    return float(np.max(drawdowns))


def calculate_sharpe_ratio(returns, periods_per_year=252):
    values = np.array(returns, dtype=np.float32)
    if len(values) < 2:
        return 0.0
    std = float(np.std(values))
    if std < 1e-8:
        return 0.0
    return float((np.mean(values) / std) * np.sqrt(periods_per_year))


def calculate_performance_metrics(portfolio_values, trade_profits=None):
    values = np.array(portfolio_values, dtype=np.float32)
    if len(values) == 0:
        return {
            'total_return_pct': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown_pct': 0.0,
            'win_rate': 0.0,
            'trade_count': 0,
            'avg_trade_profit': 0.0,
        }

    total_return_pct = float((values[-1] / max(values[0], 1e-8) - 1.0) * 100.0)
    returns = np.diff(values) / np.maximum(values[:-1], 1e-8) if len(values) > 1 else np.array([0.0], dtype=np.float32)

    trade_profits = trade_profits or []
    wins = [p for p in trade_profits if p > 0]

    return {
        'total_return_pct': total_return_pct,
        'sharpe_ratio': calculate_sharpe_ratio(returns),
        'max_drawdown_pct': calculate_max_drawdown(values) * 100.0,
        'win_rate': float(len(wins) / len(trade_profits)) if trade_profits else 0.0,
        'trade_count': len(trade_profits),
        'avg_trade_profit': float(np.mean(trade_profits)) if trade_profits else 0.0,
        'trade_frequency': float(len(trade_profits) / max(len(values) - 1, 1)),
    }


def save_metrics_csv(metrics_history, path):
    df = pd.DataFrame(metrics_history)
    df.to_csv(path, index=False)
    return df
