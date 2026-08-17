import numpy as np


def _compute_volume_profile_levels(df, lookback):
    lookback = max(int(lookback), 5)
    rolling_low = df['low'].rolling(window=lookback, min_periods=lookback).min()
    rolling_high = df['high'].rolling(window=lookback, min_periods=lookback).max()
    total_volume = df['volume'].rolling(window=lookback, min_periods=lookback).sum()
    vwap_num = (df['close'] * df['volume']).rolling(window=lookback, min_periods=lookback).sum()
    poc = vwap_num / total_volume.replace(0, np.nan)
    value_range = rolling_high - rolling_low
    vah = poc + 0.35 * value_range
    val = poc - 0.35 * value_range
    return poc, vah, val


def add_indicators(df, indicators=None, volume_profile_lookback=30):
    """Add technical indicators to the dataframe."""
    df = df.copy()
    indicators = {name.lower() for name in (indicators or [])}
    include_all = not indicators or 'all' in indicators

    if include_all or 'sma' in indicators:
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()

    if include_all or 'ema' in indicators:
        df['ema_10'] = df['close'].ewm(span=10).mean()

    if include_all or 'rsi' in indicators:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))

    if include_all or 'macd' in indicators:
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['signal_line'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['signal_line']

    if include_all or 'bollinger' in indicators:
        df['bb_mid'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_mid'] + (bb_std * 2)
        df['bb_lower'] = df['bb_mid'] - (bb_std * 2)

    if include_all or 'volume_sma' in indicators:
        df['volume_sma'] = df['volume'].rolling(window=20).mean()

    if include_all or 'volume_profile' in indicators:
        poc, vah, val = _compute_volume_profile_levels(df, lookback=volume_profile_lookback)
        df['vp_poc'] = poc
        df['vp_vah'] = vah
        df['vp_val'] = val
        df['dist_poc'] = (df['close'] - poc) / poc.replace(0, np.nan)
        df['dist_vah'] = (df['close'] - vah) / vah.replace(0, np.nan)
        df['dist_val'] = (df['close'] - val) / val.replace(0, np.nan)

    return df.replace([np.inf, -np.inf], np.nan).dropna()
