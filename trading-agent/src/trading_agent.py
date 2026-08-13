"""
Trading Agent Environment with Volume Profile Analysis
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
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
        self.bin_edges = None
        self.bin_centers = None
        
    def calculate_profile(self) -> Dict:
        """Calculate volume profile and key levels"""
        if len(self.price_data) == 0 or len(self.volume_data) == 0:
            return {'poc': 0.0, 'vah': 0.0, 'val': 0.0, 'profile': np.zeros(self.bins), 'bins': np.array([])}

        min_price = float(np.min(self.price_data))
        max_price = float(np.max(self.price_data))
        if min_price == max_price:
            # Degenerate case: all prices identical
            profile = np.zeros(self.bins)
            profile[0] = np.sum(self.volume_data)
            bins = np.linspace(min_price, max_price + 1e-6, self.bins + 1)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            self.profile = profile
            self.bin_edges = bins
            self.bin_centers = bin_centers
            self.poc = bin_centers[0]
            self.vah = bin_centers[0]
            self.val = bin_centers[0]
            return {'poc': self.poc, 'vah': self.vah, 'val': self.val, 'profile': profile, 'bins': bin_centers}
        
        # Create price bin edges (bins+1 edges) and bin centers
        edges = np.linspace(min_price, max_price, self.bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        bin_indices = np.digitize(self.price_data, edges) - 1  # shift to 0-based
        
        # Aggregate volume by price bin
        profile = np.zeros(self.bins)
        for i, vol in enumerate(self.volume_data):
            idx = int(bin_indices[i])
            if 0 <= idx < self.bins:
                profile[idx] += vol
            else:
                # Clamp out-of-range indices to nearest bin
                if idx < 0:
                    profile[0] += vol
                elif idx >= self.bins:
                    profile[-1] += vol
        
        self.profile = profile
        self.bin_edges = edges
        self.bin_centers = centers
        
        # Calculate Point of Control (POC)
        poc_idx = int(np.argmax(profile))
        self.poc = centers[poc_idx]
        
        # Calculate Value Area (70% of volume)
        sorted_indices = np.argsort(profile)[::-1]
        cumsum = 0.0
        total_vol = float(np.sum(profile))
        target_vol = total_vol * 0.70
        
        value_area_indices = []
        for idx in sorted_indices:
            cumsum += profile[int(idx)]
            value_area_indices.append(int(idx))
            if cumsum >= target_vol:
                break
        
        if len(value_area_indices) == 0:
            self.vah = self.poc
            self.val = self.poc
        else:
            value_area_prices = centers[value_area_indices]
            self.vah = float(np.max(value_area_prices))
            self.val = float(np.min(value_area_prices))
        
        return {
            'poc': self.poc,
            'vah': self.vah,
            'val': self.val,
            'profile': profile,
            'bins': centers
        }


class TradingEnvironmentWithVolumeProfile:
    """Stock trading environment using volume profile for targets"""
    
    def __init__(self, stock_symbol: str, initial_balance: float = 10000, lookback_days: int = 30):
        self.stock_symbol = stock_symbol
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.shares_held = 0
        self.entry_price: Optional[float] = None
        self.trades = []
        self.current_step = 0
        self.prices: np.ndarray = np.array([])
        self.volumes: np.ndarray = np.array([])
        self.lookback_days = lookback_days
        
        # Initialize volume profile levels to 0 instead of None
        self.poc = 0.0
        self.vah = 0.0
        self.val = 0.0
        
    def load_data(self, start_date: str, end_date: str):
        """Load historical stock data"""
        data = yf.download(self.stock_symbol, start=start_date, end=end_date, progress=False)
        if data is None or data.empty:
            self.prices = np.array([])
            self.volumes = np.array([])
            return self.prices, self.volumes
        self.prices = data['Close'].values
        self.volumes = data['Volume'].values
        return self.prices, self.volumes
    
    def update_volume_profile(self, lookback_window: int = None):
        """Update volume profile based on historical data"""
        if lookback_window is None:
            lookback_window = min(self.lookback_days, max(2, self.current_step))
        else:
            lookback_window = max(2, lookback_window)
        
        if len(self.prices) == 0:
            return
        
        start_idx = max(0, int(self.current_step - lookback_window))
        end_idx = int(min(self.current_step, len(self.prices)))
        
        if start_idx >= end_idx:
            return
        
        prices = self.prices[start_idx:end_idx]
        volumes = self.volumes[start_idx:end_idx]
        
        analyzer = VolumeProfileAnalyzer(prices, volumes, bins=50)
        profile = analyzer.calculate_profile()
        
        self.poc = profile.get('poc', 0.0) or 0.0
        self.vah = profile.get('vah', 0.0) or 0.0
        self.val = profile.get('val', 0.0) or 0.0
    
    def reset(self):
        """Reset environment to initial state"""
        self.balance = self.initial_balance
        self.shares_held = 0
        self.entry_price = None
        self.trades = []
        # Start at lookback_days or last available index if less data
        if len(self.prices) > 0:
            self.current_step = int(min(self.lookback_days, max(0, len(self.prices) - 1)))
        else:
            self.current_step = 0
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state vector"""
        # Ensure current_step in bounds
        if len(self.prices) == 0:
            # Empty state
            return np.zeros(6, dtype=np.float32)
        self.current_step = int(min(self.current_step, len(self.prices) - 1))

        self.update_volume_profile()
        
        current_price = float(self.prices[self.current_step])
        portfolio_value = self.balance + (self.shares_held * current_price)
        balance_ratio = self.balance / (self.initial_balance + 1e-8)
        
        # Safe division with default values if levels are 0
        poc_dist = (current_price - self.poc) / (self.poc + 1e-8) if self.poc != 0 else 0.0
        vah_dist = (current_price - self.vah) / (self.vah + 1e-8) if self.vah != 0 else 0.0
        val_dist = (current_price - self.val) / (self.val + 1e-8) if self.val != 0 else 0.0
        
        recent_start = max(0, int(self.current_step - 100))
        recent_prices = self.prices[recent_start:self.current_step + 1]
        price_norm = float(current_price / (np.max(recent_prices) + 1e-8)) if len(recent_prices) > 0 else 0.0
        
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
        if len(self.prices) == 0:
            return self._get_state(), 0.0, True

        self.current_step = int(min(self.current_step, len(self.prices) - 1))
        current_price = float(self.prices[self.current_step])
        reward = -0.01
        
        if action == 1:  # Buy
            max_shares = int(self.balance / (current_price + 1e-8))
            if max_shares > 0:
                distance_to_val = abs(current_price - self.val) / (self.val + 1e-8) if self.val != 0 else 1
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
                
                if self.entry_price:
                    profit = revenue - (self.entry_price * self.shares_held)
                    profit_pct = profit / (self.entry_price * self.shares_held + 1e-8)
                else:
                    profit = 0
                    profit_pct = 0
                
                distance_to_vah = abs(current_price - self.vah) / (self.vah + 1e-8) if self.vah != 0 else 1
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
        
        # advance
        self.current_step += 1
        done = self.current_step >= len(self.prices)
        
        if done and self.shares_held > 0:
            final_idx = min(self.current_step - 1, len(self.prices) - 1)
            final_price = float(self.prices[final_idx])
            self.balance += self.shares_held * final_price
            self.shares_held = 0
        
        next_idx = min(self.current_step, len(self.prices) - 1)
        next_price = float(self.prices[next_idx])
        portfolio_value = self.balance + (self.shares_held * next_price)
        reward += (portfolio_value - self.initial_balance) / (self.initial_balance + 1e-8) * 0.1
        
        return self._get_state(), float(reward), done
