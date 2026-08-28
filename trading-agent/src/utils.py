"""
Utility functions for trading agent
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float:
    """Calculate annualized Sharpe ratio

    returns: list of periodic returns (e.g., daily returns)
    risk_free_rate: annual risk-free rate
    periods_per_year: number of periods per year for annualization
    """
    if len(returns) == 0:
        return 0.0

    # Convert to numpy array
    rets = np.array(returns)
    # Convert annual risk-free to periodic
    rf_periodic = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess_returns = rets - rf_periodic
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns) + 1e-8
    # Annualize
    sharpe = (mean_excess / std_excess) * np.sqrt(periods_per_year)
    return float(sharpe)


def calculate_max_drawdown(portfolio_values: List[float]) -> float:
    """Calculate maximum drawdown (as positive number)

    Returns the maximum drawdown (negative or zero). Uses portfolio_values as time series.
    """
    if len(portfolio_values) == 0:
        return 0.0

    pv = np.array(portfolio_values, dtype=float)
    running_max = np.maximum.accumulate(pv)
    drawdowns = (pv - running_max) / (running_max + 1e-8)
    return float(np.min(drawdowns))


def calculate_sortino_ratio(returns: List[float], target_return: float = 0.0, risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float:
    """Calculate annualized Sortino ratio"""
    if len(returns) == 0:
        return 0.0

    rets = np.array(returns)
    rf_periodic = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = rets - rf_periodic
    downside = excess[excess < target_return]
    if len(downside) == 0:
        return float(np.inf) if np.mean(excess) > 0 else 0.0
    downside_std = np.std(downside) + 1e-8
    sortino = (np.mean(excess) / downside_std) * np.sqrt(periods_per_year)
    return float(sortino)


def normalize_state(state: np.ndarray, min_vals: np.ndarray = None, max_vals: np.ndarray = None) -> np.ndarray:
    """Normalize state to [-1, 1] range"""
    state = np.asarray(state, dtype=float)
    if min_vals is None or max_vals is None:
        mean = np.mean(state)
        std = np.std(state) + 1e-8
        return (state - mean) / std

    min_vals = np.asarray(min_vals, dtype=float)
    max_vals = np.asarray(max_vals, dtype=float)
    return 2 * (state - min_vals) / (max_vals - min_vals + 1e-8) - 1


def calculate_trade_metrics(trades: List[Dict]) -> Dict:
    """Calculate metrics from trade list"""
    if len(trades) == 0:
        return {
            'total_trades': 0,
            'buy_trades': 0,
            'sell_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0
        }

    buy_trades = [t for t in trades if t['type'] == 'BUY']
    sell_trades = [t for t in trades if t['type'] == 'SELL']

    winning = [t for t in sell_trades if t.get('profit_pct', 0) > 0]
    losing = [t for t in sell_trades if t.get('profit_pct', 0) <= 0]

    win_rate = (len(winning) / len(sell_trades) * 100) if len(sell_trades) > 0 else 0
    avg_win = float(np.mean([t.get('profit_pct', 0) for t in winning])) if len(winning) > 0 else 0.0
    avg_loss = float(np.mean([t.get('profit_pct', 0) for t in losing])) if len(losing) > 0 else 0.0

    # Calculate profit factor, handle edge cases
    if len(losing) > 0:
        pf = abs(avg_win * len(winning) / (avg_loss * len(losing) + 1e-8))
        profit_factor = pf if not np.isinf(pf) else float('inf')
    else:
        # No losing trades: either 0 trades or all winners
        profit_factor = float('inf') if len(winning) > 0 else 0.0

    return {
        'total_trades': len(trades),
        'buy_trades': len(buy_trades),
        'sell_trades': len(sell_trades),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor
    }


def calculate_activity_metrics(trades: List[Dict], prices: np.ndarray, initial_balance: float, portfolio_history: List[float] = None) -> Dict:
    """Calculate trade activity metrics: trades per day, max exposure, avg hold duration.

    Args:
        trades: list of trade dicts (each has 'type', 'step', 'shares', 'price')
        prices: full price array from the environment
        initial_balance: initial portfolio value (used to compute exposure %)
        portfolio_history: portfolio value at each step (used to compute exposure vs current portfolio)
    """
    if not trades or len(prices) == 0:
        return {
            'trades_per_day': 0.0,
            'max_exposure_pct': 0.0,
            'avg_hold_days': 0.0,
        }

    total_days = max(len(prices), 1)
    trades_per_day = len(trades) / total_days

    # Max exposure: highest (shares * price) relative to the portfolio value at that step.
    # Using initial_balance as fallback when portfolio_history is unavailable.
    max_exposure_pct = 0.0
    for t in trades:
        if t['type'] != 'BUY':
            continue
        exposure = float(t.get('shares', 0)) * float(t.get('price', 0))
        step = int(t.get('step', 0))
        if portfolio_history and step < len(portfolio_history):
            portfolio_at_step = float(portfolio_history[step]) or initial_balance
        else:
            portfolio_at_step = initial_balance
        pct = (exposure / (portfolio_at_step + 1e-8)) * 100.0
        if pct > max_exposure_pct:
            max_exposure_pct = pct

    # Avg hold duration: match each BUY to the next SELL by step order
    hold_durations = []
    buy_step = None
    for t in trades:
        if t['type'] == 'BUY':
            buy_step = int(t['step'])
        elif t['type'] == 'SELL' and buy_step is not None:
            hold_durations.append(int(t['step']) - buy_step)
            buy_step = None

    avg_hold_days = float(np.mean(hold_durations)) if hold_durations else 0.0

    return {
        'trades_per_day': round(trades_per_day, 4),
        'max_exposure_pct': round(max_exposure_pct, 2),
        'avg_hold_days': round(avg_hold_days, 2),
    }
