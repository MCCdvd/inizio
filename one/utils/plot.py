import matplotlib.pyplot as plt
import numpy as np

def plot_rewards(rewards_history):
    """Plot training rewards history."""
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards_history, alpha=0.6, label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards Over Episodes')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    window = 10
    moving_avg = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
    plt.plot(moving_avg, color='red', linewidth=2, label=f'{window}-Episode MA')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Moving Average of Rewards')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_training_metrics(metrics_history):
    """Plot comprehensive training metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Training Performance Metrics', fontsize=16, fontweight='bold')
    
    episodes = metrics_history['episode']
    
    # 1. Total Return %
    ax = axes[0, 0]
    ax.plot(episodes, metrics_history['total_return_pct'], 'b-', alpha=0.6, label='Return')
    window = 5
    if len(episodes) >= window:
        ma = np.convolve(metrics_history['total_return_pct'], np.ones(window)/window, mode='valid')
        ax.plot(episodes[window-1:], ma, 'r-', linewidth=2, label=f'{window}-Episode MA')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return (%)')
    ax.set_title('Total Return per Episode')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. Portfolio Value
    ax = axes[0, 1]
    ax.plot(episodes, metrics_history['final_portfolio_value'], 'g-', alpha=0.6)
    ax.axhline(y=10000, color='k', linestyle='--', alpha=0.3, label='Initial Balance')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Portfolio Value ($)')
    ax.set_title('Final Portfolio Value per Episode')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 3. Max Drawdown %
    ax = axes[0, 2]
    ax.plot(episodes, metrics_history['max_drawdown_pct'], 'r-', alpha=0.6)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Max Drawdown (%)')
    ax.set_title('Maximum Drawdown per Episode')
    ax.grid(True, alpha=0.3)
    
    # 4. Epsilon Decay
    ax = axes[1, 0]
    ax.plot(episodes, metrics_history['epsilon'], 'purple', alpha=0.7, marker='o', markersize=3)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Epsilon (Exploration Rate)')
    ax.set_title('Exploration Rate Decay')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # 5. Number of Trades
    ax = axes[1, 1]
    ax.bar(episodes, metrics_history['trades_count'], alpha=0.7, color='orange')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Number of Trades')
    ax.set_title('Trading Activity per Episode')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 6. Realized P&L
    ax = axes[1, 2]
    colors = ['green' if x >= 0 else 'red' for x in metrics_history['realized_pnl']]
    ax.bar(episodes, metrics_history['realized_pnl'], color=colors, alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Realized P&L ($)')
    ax.set_title('Realized Profit/Loss per Episode')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
