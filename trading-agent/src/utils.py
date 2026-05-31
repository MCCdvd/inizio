"""
Utility functions for trading agent
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio"""
    if len(returns) == 0:
        return 0
    
    excess_returns = np.array(returns) - risk_free_rate
    return np.mean(excess_returns) / (np.std(excess_returns) + 1e-8)


def calculate_max_drawdown(portfolio_values: List[float]) -> float:
    """Calculate maximum drawdown"""
    if len(portfolio_values) == 0:
        return 0
    
    running_max = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - running_max) / running_max
    return np.min(drawdown)


def calculate_sortino_ratio(returns: List[float], target_return: float = 0, risk_free_rate: float = 0.02) -> float:
    """Calculate Sortino ratio"""
    if len(returns) == 0:
        return 0
    
    excess_returns = np.array(returns) - risk_free_rate
    downside_returns = np.where(excess_returns < target_return, excess_returns, 0)
    
    return np.mean(excess_returns) / (np.std(downside_returns) + 1e-8)


def normalize_state(state: np.ndarray, min_vals: np.ndarray = None, max_vals: np.ndarray = None) -> np.ndarray:
    """Normalize state to [-1, 1] range"""
    if min_vals is None or max_vals is None:
        return (state - np.mean(state)) / (np.std(state) + 1e-8)
    
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
    
    win_rate = len(winning) / len(sell_trades) * 100 if len(sell_trades) > 0 else 0
    avg_win = np.mean([t.get('profit_pct', 0) for t in winning]) if len(winning) > 0 else 0
    avg_loss = np.mean([t.get('profit_pct', 0) for t in losing]) if len(losing) > 0 else 0
    
    profit_factor = abs(avg_win * len(winning) / (avg_loss * len(losing) + 1e-8))
    
    return {
        'total_trades': len(trades),
        'buy_trades': len(buy_trades),
        'sell_trades': len(sell_trades),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor
    }
