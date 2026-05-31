"""
Trading Agent Environment with Volume Profile Analysis
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
import yfinance as yf
from datetime import datetime, timedelta


class VolumeProfileAnalyzer:
    """Calculate volume profile and identify key levels"""
    
    def __init__(self, price_data: np.ndarray, volume_data: np.ndarray, bins: int = 100):
        self.price_data = price_data
        self.volume_data = volume_data
        self.bins = bins
        self.profile = None
        self.poc = None
        self.vah = None
        self.val = None
        
    def calculate_profile(self) -> Dict:
        """Calculate volume profile and key levels"""
        min_price = np.min(self.price_data)
        max_price = np.max(self.price_data)
        
        # Create price bins
        bins = np.linspace(min_price, max_price, self.bins)
        bin_indices = np.digitize(self.price_data, bins)
        
        # Aggregate volume by price bin
        profile = np.zeros(self.bins)
        for i, vol in enumerate(self.volume_data):
            if 0 <= bin_indices[i] < self.bins:
                profile[bin_indices[i]] += vol
        
        self.profile = profile
        self.bin_edges = bins
        
        # Calculate Point of Control (POC)
        poc_idx = np.argmax(profile)
        self.poc = bins[poc_idx]
        
        # Calculate Value Area (70% of volume)
        sorted_indices = np.argsort(profile)[::-1]
        cumsum = 0
        total_vol = np.sum(profile)
        target_vol = total_vol * 0.70
        
        value_area_indices = []
        for idx in sorted_indices:
            cumsum += profile[idx]
            value_area_indices.append(idx)
            if cumsum >= target_vol:
                break
        
        value_area_prices = bins[value_area_indices]
        self.vah = np.max(value_area_prices)
        self.val = np.min(value_area_prices)
        
        return {
            'poc': self.poc,
            'vah': self.vah,
            'val': self.val,
            'profile': profile,
            'bins': bins
        }


class TradingEnvironmentWithVolumeProfile:
    """Stock trading environment using volume profile for targets"""
    
    def __init__(self, stock_symbol: str, initial_balance: float = 10000, lookback_days: int = 30):
        self.stock_symbol = stock_symbol
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.shares_held = 0
        self.entry_price = None
        self.trades = []
        self.current_step = 0
        self.prices = []
        self.volumes = []
        self.lookback_days = lookback_days
        
        # Initialize volume profile levels to 0 instead of None
        self.poc = 0.0
        self.vah = 0.0
        self.val = 0.0
        
    def load_data(self, start_date: str, end_date: str):
        """Load historical stock data"""
        data = yf.download(self.stock_symbol, start=start_date, end=end_date, progress=False)
        self.prices = data['Close'].values
        self.volumes = data['Volume'].values
        return self.prices, self.volumes
    
    def update_volume_profile(self, lookback_window: int = None):
        """Update volume profile based on historical data"""
        if lookback_window is None:
            lookback_window = min(self.lookback_days, self.current_step)
        
        start_idx = max(0, self.current_step - lookback_window)
        end_idx = self.current_step
        
        if start_idx >= end_idx:
            return
        
        prices = self.prices[start_idx:end_idx]
        volumes = self.volumes[start_idx:end_idx]
        
        analyzer = VolumeProfileAnalyzer(prices, volumes, bins=50)
        profile = analyzer.calculate_profile()
        
        self.poc = profile['poc']
        self.vah = profile['vah']
        self.val = profile['val']
    
    def reset(self):
        """Reset environment to initial state"""
        self.balance = self.initial_balance
        self.shares_held = 0
        self.entry_price = None
        self.trades = []
        self.current_step = self.lookback_days
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state vector"""
        self.update_volume_profile()
        
        current_price = self.prices[self.current_step]
        portfolio_value = self.balance + (self.shares_held * current_price)
        balance_ratio = self.balance / self.initial_balance
        
        # Safe division with default values if levels are 0
        poc_dist = (current_price - self.poc) / self.poc if self.poc > 0 else 0.0
        vah_dist = (current_price - self.vah) / self.vah if self.vah > 0 else 0.0
        val_dist = (current_price - self.val) / self.val if self.val > 0 else 0.0
        
        recent_prices = self.prices[max(0, self.current_step - 100):self.current_step]
        price_norm = current_price / np.max(recent_prices) if len(recent_prices) > 0 else 0.0
        
        state = np.array([
            float(balance_ratio),
            float(self.shares_held / 100),
            float(poc_dist),
            float(vah_dist),
            float(val_dist),
            float(price_norm)
        ], dtype=np.float32)
        
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Execute trading action"""
        current_price = self.prices[self.current_step]
        reward = -0.01
        
        if action == 1:  # Buy
            max_shares = int(self.balance / current_price)
            if max_shares > 0:
                distance_to_val = abs(current_price - self.val) / self.val if self.val > 0 else 1
                buy_incentive = max(0, 0.05 - distance_to_val)
                
                shares_to_buy = max(1, int(max_shares * (0.5 + buy_incentive)))
                shares_to_buy = min(shares_to_buy, max_shares)
                
                cost = shares_to_buy * current_price
                self.balance -= cost
                self.shares_held += shares_to_buy
                self.entry_price = current_price
                
                self.trades.append({
                    'type': 'BUY',
                    'step': self.current_step,
                    'price': current_price,
                    'shares': shares_to_buy,
                    'poc': self.poc,
                    'val': self.val,
                    'vah': self.vah
                })
                reward += 0.1
        
        elif action == 2:  # Sell
            if self.shares_held > 0:
                revenue = self.shares_held * current_price
                self.balance += revenue
                
                profit = revenue - (self.entry_price * self.shares_held) if self.entry_price else 0
                profit_pct = profit / (self.entry_price * self.shares_held) if self.entry_price else 0
                
                distance_to_vah = abs(current_price - self.vah) / self.vah if self.vah > 0 else 1
                sell_incentive = max(0, 0.05 - distance_to_vah)
                
                self.trades.append({
                    'type': 'SELL',
                    'step': self.current_step,
                    'price': current_price,
                    'shares': self.shares_held,
                    'profit_pct': profit_pct,
                    'poc': self.poc,
                    'val': self.val,
                    'vah': self.vah
                })
                
                reward += profit_pct + sell_incentive
                self.shares_held = 0
                self.entry_price = None
        
        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1
        
        if done and self.shares_held > 0:
            final_price = self.prices[self.current_step - 1]
            self.balance += self.shares_held * final_price
            self.shares_held = 0
        
        next_price = self.prices[self.current_step] if self.current_step < len(self.prices) else self.prices[-1]
        portfolio_value = self.balance + (self.shares_held * next_price)
        reward += (portfolio_value - self.initial_balance) / self.initial_balance * 0.1
        
        return self._get_state(), reward, done
