import matplotlib.pyplot as plt
import numpy as np

def plot_rewards(rewards_history):
    """Plot training rewards history."""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards_history, alpha=0.6)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards Over Episodes')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    window = 10
    moving_avg = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
    plt.plot(moving_avg, color='red', label=f'{window}-Episode Moving Average')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Moving Average of Rewards')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
