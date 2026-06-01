import pandas as pd
import numpy as np

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=window).mean()
    avg_loss = pd.Series(loss).rolling(window=window).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_macd(close, a=12, b=26, signal=9):
    ema_fast = close.ewm(span=a, min_periods=a).mean()
    ema_slow = close.ewm(span=b, min_periods=b).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, min_periods=signal).mean()
    return macd, signal_line

def add_indicators(df):
    df['rsi'] = compute_rsi(df['close'])
    macd, macd_signal = compute_macd(df['close'])
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['ma50'] = df['close'].rolling(window=50).mean()
    df['volatility'] = df['close'].rolling(window=20).std()
    # Assume value profile columns are already in df (poc, vah, val)
    df['poc_distance'] = (df['close'] - df['poc']) / df['poc']
    df['vah_distance'] = (df['close'] - df['vah']) / df['vah']
    df['val_distance'] = (df['close'] - df['val']) / df['val']
    df['price_norm'] = df['close'] / df['close'].rolling(window=50).max()
    return df
