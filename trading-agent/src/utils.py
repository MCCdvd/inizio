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

    profit_factor = abs(avg_win * len(winning) / (avg_loss * len(losing) + 1e-8)) if (len(losing) > 0) else float('inf')

    return {
        'total_trades': len(trades),
        'buy_trades': len(buy_trades),
        'sell_trades': len(sell_trades),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor
    }
