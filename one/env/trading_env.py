import numpy as np
import pandas as pd


class TradingEnv:
    """Trading environment with indicator-rich state and shaped rewards."""

    def __init__(
        self,
        df,
        initial_balance=10000.0,
        window_size=30,
        drawdown_penalty=0.1,
        trade_bonus=0.05,
        hold_penalty=0.001,
        risk_reward_weight=0.05,
        max_hold_steps=20,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.initial_balance = float(initial_balance)
        self.window_size = int(window_size)
        self.drawdown_penalty = float(drawdown_penalty)
        self.trade_bonus = float(trade_bonus)
        self.hold_penalty = float(hold_penalty)
        self.risk_reward_weight = float(risk_reward_weight)
        self.max_hold_steps = int(max_hold_steps)

        self._prepare_features()
        self.reset()

    def _prepare_features(self):
        numeric = self.df.select_dtypes(include=[np.number]).copy()
        if numeric.empty:
            raise ValueError("TradingEnv requires numeric OHLCV/indicator columns")

        self.feature_cols = numeric.columns.tolist()
        self.feature_values = numeric.values.astype(np.float32)
        self.feature_mean = np.nanmean(self.feature_values, axis=0)
        self.feature_std = np.nanstd(self.feature_values, axis=0)
        self.feature_std[self.feature_std < 1e-8] = 1.0

        self.price_col = 'close' if 'close' in self.df.columns else self.feature_cols[0]

        base_state = self.window_size * len(self.feature_cols)
        self.portfolio_feature_count = 6
        self.state_size = base_state + self.portfolio_feature_count

    def reset(self):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.entry_price = 0.0
        self.holding_steps = 0

        self.peak_portfolio_value = self.initial_balance
        self.portfolio_history = [self.initial_balance]
        self.step_returns = []
        self.trades = []
        self.trade_profits = []
        self.realized_pnl = 0.0

        return self._get_state()

    def _current_price(self):
        return float(self.df.loc[self.current_step, self.price_col])

    def _portfolio_value(self, price=None):
        if price is None:
            price = self._current_price()
        return float(self.balance + self.shares_held * price)

    def _get_state(self):
        start = self.current_step - self.window_size
        end = self.current_step

        if start < 0:
            return np.zeros(self.state_size, dtype=np.float32)

        window = self.feature_values[start:end]
        norm_window = (window - self.feature_mean) / self.feature_std
        norm_window = np.clip(norm_window, -5.0, 5.0)

        price = self._current_price()
        portfolio_value = self._portfolio_value(price)
        self.peak_portfolio_value = max(self.peak_portfolio_value, portfolio_value)
        drawdown = (self.peak_portfolio_value - portfolio_value) / max(self.peak_portfolio_value, 1e-8)

        cash_ratio = self.balance / max(portfolio_value, 1e-8)
        position_ratio = (self.shares_held * price) / max(portfolio_value, 1e-8)
        position_flag = 1.0 if self.shares_held > 0 else 0.0
        unrealized_return = 0.0
        if self.shares_held > 0 and self.entry_price > 0:
            unrealized_return = (price - self.entry_price) / self.entry_price
        step_progress = self.current_step / max(len(self.df) - 1, 1)
        hold_ratio = self.holding_steps / max(self.max_hold_steps, 1)

        portfolio_features = np.array(
            [
                cash_ratio,
                position_ratio,
                position_flag,
                unrealized_return,
                drawdown,
                min(hold_ratio, 2.0),
            ],
            dtype=np.float32,
        )

        state = np.concatenate([norm_window.flatten().astype(np.float32), portfolio_features])
        return state

    def _risk_adjusted_component(self):
        if len(self.step_returns) < 2:
            return 0.0
        window = np.array(self.step_returns[-20:], dtype=np.float32)
        volatility = float(np.std(window))
        if volatility < 1e-8:
            return 0.0
        return float(np.mean(window) / volatility)

    def _episode_metrics(self, final_price):
        portfolio_values = np.array(self.portfolio_history, dtype=np.float32)
        if len(portfolio_values) == 0:
            portfolio_values = np.array([self.initial_balance], dtype=np.float32)

        running_peak = np.maximum.accumulate(portfolio_values)
        drawdowns = (running_peak - portfolio_values) / np.maximum(running_peak, 1e-8)
        max_drawdown = float(np.max(drawdowns)) if len(drawdowns) else 0.0

        final_value = self._portfolio_value(final_price)
        total_return_pct = ((final_value / self.initial_balance) - 1.0) * 100.0

        returns = np.array(self.step_returns, dtype=np.float32)
        sharpe = 0.0
        if len(returns) > 1:
            std = float(np.std(returns))
            if std > 1e-8:
                sharpe = float((np.mean(returns) / std) * np.sqrt(252.0))

        winning = [p for p in self.trade_profits if p > 0]
        win_rate = float(len(winning) / len(self.trade_profits)) if self.trade_profits else 0.0
        avg_trade_profit = float(np.mean(self.trade_profits)) if self.trade_profits else 0.0

        return {
            'final_portfolio_value': final_value,
            'total_return_pct': float(total_return_pct),
            'max_drawdown_pct': float(max_drawdown * 100.0),
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'trade_count': len(self.trade_profits),
            'trade_frequency': float(len(self.trade_profits) / max(self.current_step - self.window_size, 1)),
            'avg_trade_profit': avg_trade_profit,
            'realized_pnl': float(self.realized_pnl),
            'episode_length': int(self.current_step - self.window_size),
        }

    def step(self, action):
        """Execute one trading action. action: 0=hold, 1=buy, 2=sell"""
        if self.current_step >= len(self.df) - 1:
            metrics = self._episode_metrics(self._current_price())
            return self._get_state(), 0.0, True, {'metrics': metrics}

        current_price = self._current_price()
        prev_value = self._portfolio_value(current_price)
        reward = 0.0

        if action == 1:  # Buy
            if self.shares_held <= 0 and self.balance > current_price:
                shares = self.balance / current_price
                self.shares_held = shares
                self.balance = 0.0
                self.entry_price = current_price
                self.holding_steps = 0
                self.trades.append({'step': self.current_step, 'action': 'buy', 'price': current_price})

        elif action == 2:  # Sell
            if self.shares_held > 0:
                proceeds = self.shares_held * current_price
                cost_basis = self.shares_held * self.entry_price
                profit = proceeds - cost_basis
                profit_pct = profit / max(cost_basis, 1e-8)

                self.balance = proceeds
                self.shares_held = 0.0
                self.realized_pnl += profit
                self.trade_profits.append(float(profit_pct))
                self.trades.append(
                    {
                        'step': self.current_step,
                        'action': 'sell',
                        'price': current_price,
                        'profit_pct': float(profit_pct),
                    }
                )
                reward += self.trade_bonus * profit_pct
                self.holding_steps = 0

        if self.shares_held > 0:
            self.holding_steps += 1

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        next_price = self._current_price()

        portfolio_value = self._portfolio_value(next_price)
        self.portfolio_history.append(portfolio_value)
        self.peak_portfolio_value = max(self.peak_portfolio_value, portfolio_value)

        step_return = (portfolio_value - prev_value) / max(prev_value, 1e-8)
        self.step_returns.append(float(step_return))
        reward += step_return

        drawdown = (self.peak_portfolio_value - portfolio_value) / max(self.peak_portfolio_value, 1e-8)
        reward -= self.drawdown_penalty * drawdown

        reward += self.risk_reward_weight * self._risk_adjusted_component()

        if self.shares_held > 0 and self.holding_steps > self.max_hold_steps and next_price <= self.entry_price:
            reward -= self.hold_penalty * (self.holding_steps - self.max_hold_steps)

        info = {
            'portfolio_value': portfolio_value,
            'step_return': float(step_return),
            'drawdown': float(drawdown),
            'trade_count': len(self.trade_profits),
        }

        if done:
            info['metrics'] = self._episode_metrics(next_price)

        next_state = self._get_state()
        return next_state, float(reward), done, info
