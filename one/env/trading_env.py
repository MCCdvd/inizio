import numpy as np
import pandas as pd

class TradingEnv:
    """Simple trading environment for RL agents."""
    
    def __init__(self, df, initial_balance=10000, window_size=30):
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.current_step = window_size
        self.balance = initial_balance
        self.shares_held = 0
        self.entry_price = 0
        self.portfolio_history = []
        
    def reset(self):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = 0
        self.entry_price = 0
        self.portfolio_history = []
        return self._get_state()
    
    def _get_state(self):
        """Get current state as feature vector."""
        if self.current_step < self.window_size:
            return np.zeros(self.window_size * 5 + 3)
        
        window = self.df.iloc[self.current_step - self.window_size:self.current_step]
        state = window[['open', 'high', 'low', 'close', 'volume']].values.flatten()
        state = np.append(state, [self.balance, self.shares_held, self.current_step / len(self.df)])
        return state.astype(np.float32)
    
    def step(self, action):
        """Execute one trading action. action: 0=hold, 1=buy, 2=sell"""
        current_price = self.df.loc[self.current_step, 'close']
        reward = 0
        done = False
        
        if action == 1:  # Buy
            if self.balance > current_price:
                shares = self.balance / current_price
                self.shares_held += shares
                self.balance = 0
                self.entry_price = current_price
                reward = 0
        
        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.balance = self.shares_held * current_price
                profit = self.balance - (self.shares_held * self.entry_price)
                reward = profit / (self.shares_held * self.entry_price) if self.entry_price > 0 else 0
                self.shares_held = 0
        
        elif action == 0:  # Hold
            if self.shares_held > 0:
                current_value = self.shares_held * current_price
                unrealized_profit = current_value - (self.shares_held * self.entry_price)
                reward = unrealized_profit / (self.shares_held * self.entry_price) if self.entry_price > 0 else 0
        
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        total_value = self.balance + (self.shares_held * current_price if self.shares_held > 0 else 0)
        self.portfolio_history.append(total_value)
        
        next_state = self._get_state()
        return next_state, reward, done, {}
