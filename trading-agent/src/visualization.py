"""
Volume Profile and Trading Analysis Visualization
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List


class VolumeProfileVisualizer:
    """Visualize volume profile with trading signals"""
    
    @staticmethod
    def plot_volume_profile(prices: np.ndarray, volumes: np.ndarray, 
                           poc: float, vah: float, val: float,
                           trades: List[Dict] = None, title: str = "Volume Profile"):
        """Plot volume profile with key levels"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Price chart with volume profile levels
        ax1.plot(prices, label='Price', linewidth=2, color='black')
        ax1.axhline(y=poc, color='blue', linestyle='--', label=f'POC: ${poc:.2f}', linewidth=2)
        ax1.axhline(y=vah, color='green', linestyle='--', label=f'VAH: ${vah:.2f}', linewidth=2)
        ax1.axhline(y=val, color='red', linestyle='--', label=f'VAL: ${val:.2f}', linewidth=2)
        
        # Plot trades
        if trades:
            buy_trades = [t for t in trades if t['type'] == 'BUY']
            sell_trades = [t for t in trades if t['type'] == 'SELL']
            
            for i, trade in enumerate(buy_trades):
                ax1.scatter(trade['step'], trade['price'], marker='^', color='green', s=100, 
                           label='Buy' if i == 0 else '')
            
            for i, trade in enumerate(sell_trades):
                ax1.scatter(trade['step'], trade['price'], marker='v', color='red', s=100,
                           label='Sell' if i == 0 else '')
        
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Price ($)')
        ax1.set_title(f'{title} - Price and Volume Profile Levels')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Volume profile histogram
        min_price = np.min(prices)
        max_price = np.max(prices)
        bins = np.linspace(min_price, max_price, 50)
        bin_indices = np.digitize(prices, bins)
        
        profile = np.zeros(len(bins))
        for i, vol in enumerate(volumes):
            if 0 <= bin_indices[i] < len(bins):
                profile[bin_indices[i]] += vol
        
        ax2.barh(bins, profile, height=(max_price - min_price) / len(bins), alpha=0.7, color='steelblue')
        ax2.axhline(y=poc, color='blue', linestyle='--', label='POC', linewidth=2)
        ax2.axhline(y=vah, color='green', linestyle='--', label='VAH', linewidth=2)
        ax2.axhline(y=val, color='red', linestyle='--', label='VAL', linewidth=2)
        
        ax2.set_xlabel('Volume')
        ax2.set_ylabel('Price ($)')
        ax2.set_title('Volume Profile Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_training_results(episode_rewards: List[float], episode_portfolios: List[float], 
                              title: str = "Training Results"):
        """Plot training metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Flatten and convert to lists to ensure 1D arrays
        episode_rewards_flat = [float(r) if hasattr(r, '__float__') else r for r in episode_rewards]
        episode_portfolios_flat = [float(p) if hasattr(p, '__float__') else p for p in episode_portfolios]
        
        # Plot rewards
        ax1.plot(episode_rewards_flat, linewidth=2, color='steelblue')
        ax1.fill_between(range(len(episode_rewards_flat)), episode_rewards_flat, alpha=0.3)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Cumulative Reward')
        ax1.set_title('Training Rewards Over Episodes')
        ax1.grid(True, alpha=0.3)
        
        # Plot returns
        returns = [(portfolio - 10000) / 10000 * 100 for portfolio in episode_portfolios_flat]
        ax2.plot(returns, linewidth=2, color='green')
        ax2.fill_between(range(len(returns)), returns, alpha=0.3, color='green')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Return (%)')
        ax2.set_title('Portfolio Returns Over Episodes')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_multi_stock_comparison(results: Dict[str, Dict]):
        """Compare multiple stock trading results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        stocks = list(results.keys())
        metrics = ['Final Portfolio', 'Total Trades', 'Win Rate', 'Total Return %']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = []
            
            for stock in stocks:
                if metric == 'Final Portfolio':
                    values.append(results[stock]['final_portfolio'])
                elif metric == 'Total Trades':
                    values.append(len(results[stock]['trades']))
                elif metric == 'Win Rate':
                    trades = results[stock]['trades']
                    winning = len([t for t in trades if t.get('profit_pct', 0) > 0])
                    win_rate = winning / len(trades) * 100 if len(trades) > 0 else 0
                    values.append(win_rate)
                elif metric == 'Total Return %':
                    values.append(results[stock]['return_pct'])
            
            colors = ['green' if v > 10000 or v > 0 else 'red' for v in values]
            ax.bar(stocks, values, color=colors, alpha=0.7)
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} by Stock')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
