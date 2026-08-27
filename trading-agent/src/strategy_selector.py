from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from agents import DQNAgent, PPOAgent, A3CAgent, get_agent
from trading_agent import TradingEnvironmentWithVolumeProfile, VolumeProfileAnalyzer
from utils import (
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_trade_metrics,
)

try:
    from sklearn.ensemble import RandomForestClassifier
except Exception:  # pragma: no cover - optional dependency fallback
    RandomForestClassifier = None


@dataclass
class RegimeFeatures:
    volatility: float
    trend_strength: float
    momentum: float
    relative_volume: float
    price_to_poc: float
    price_to_vah: float
    price_to_val: float
    value_area_width: float
    rsi: float
    bollinger_width: float

    def as_array(self) -> np.ndarray:
        return np.array([
            self.volatility,
            self.trend_strength,
            self.momentum,
            self.relative_volume,
            self.price_to_poc,
            self.price_to_vah,
            self.price_to_val,
            self.value_area_width,
            self.rsi,
            self.bollinger_width,
        ], dtype=np.float32)

    def as_dict(self) -> Dict[str, float]:
        return {
            'volatility': self.volatility,
            'trend_strength': self.trend_strength,
            'momentum': self.momentum,
            'relative_volume': self.relative_volume,
            'price_to_poc': self.price_to_poc,
            'price_to_vah': self.price_to_vah,
            'price_to_val': self.price_to_val,
            'value_area_width': self.value_area_width,
            'rsi': self.rsi,
            'bollinger_width': self.bollinger_width,
        }


class HoldStrategy:
    name = 'cash'

    def act(self, env: TradingEnvironmentWithVolumeProfile, state: np.ndarray) -> int:
        return 0


class VolumeProfileStrategy:
    name = 'volume_profile'

    def act(self, env: TradingEnvironmentWithVolumeProfile, state: np.ndarray) -> int:
        if len(env.prices) == 0:
            return 0

        idx = min(max(env.current_step, 0), len(env.prices) - 1)
        current_price = float(env.prices[idx])
        prev_price = float(env.prices[idx - 1]) if idx > 0 else current_price

        if env.shares_held == 0:
            if env.val > 0 and prev_price < env.val <= current_price:
                return 1
            if env.val > 0 and current_price <= env.val * 1.01:
                return 1
            return 0

        if env.vah > 0 and prev_price <= env.vah < current_price:
            return 2
        if env.vah > 0 and current_price >= env.vah * 0.995:
            return 2
        return 0


class MarketRegimeFeatureExtractor:
    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> float:
        return float(numerator / (denominator + 1e-8))

    @staticmethod
    def _compute_rsi(returns: np.ndarray, period: int = 14) -> float:
        if returns.size == 0:
            return 50.0
        tail = returns[-period:]
        gains = tail[tail > 0]
        losses = -tail[tail < 0]
        avg_gain = float(np.mean(gains)) if gains.size else 0.0
        avg_loss = float(np.mean(losses)) if losses.size else 0.0
        if avg_loss == 0.0:
            return 100.0 if avg_gain > 0 else 50.0
        rs = avg_gain / avg_loss
        return float(100.0 - (100.0 / (1.0 + rs)))

    @classmethod
    def extract(
        cls,
        prices: Sequence[float],
        volumes: Sequence[float],
        lookback_window: int = 30,
    ) -> RegimeFeatures:
        prices_arr = np.asarray(prices, dtype=np.float64)
        volumes_arr = np.asarray(volumes, dtype=np.float64)
        if prices_arr.size == 0 or volumes_arr.size == 0:
            return RegimeFeatures(*(0.0 for _ in range(10)))

        lb = min(max(5, int(lookback_window)), prices_arr.size)
        price_window = prices_arr[-lb:]
        volume_window = volumes_arr[-lb:]
        current_price = float(price_window[-1])
        returns = np.diff(price_window) / (price_window[:-1] + 1e-8) if price_window.size > 1 else np.array([], dtype=np.float64)
        volatility = float(np.std(returns) * np.sqrt(max(len(returns), 1)))
        momentum = cls._safe_ratio(current_price - float(price_window[0]), float(price_window[0])) if price_window.size > 1 else 0.0

        if price_window.size > 1:
            x = np.arange(price_window.size, dtype=np.float64)
            slope = float(np.polyfit(x, price_window, 1)[0])
            trend_strength = cls._safe_ratio(slope * price_window.size, np.mean(price_window))
        else:
            trend_strength = 0.0

        relative_volume = cls._safe_ratio(float(volume_window[-1]), float(np.mean(volume_window)))
        analyzer = VolumeProfileAnalyzer(price_window, volume_window, bins=min(50, max(10, price_window.size)))
        profile = analyzer.calculate_profile()
        poc = float(profile.get('poc', current_price) or current_price)
        vah = float(profile.get('vah', current_price) or current_price)
        val = float(profile.get('val', current_price) or current_price)

        price_to_poc = cls._safe_ratio(current_price - poc, poc)
        price_to_vah = cls._safe_ratio(current_price - vah, vah)
        price_to_val = cls._safe_ratio(current_price - val, val)
        value_area_width = cls._safe_ratio(vah - val, current_price)

        mean_price = float(np.mean(price_window))
        std_price = float(np.std(price_window))
        bollinger_width = cls._safe_ratio(4.0 * std_price, mean_price)
        rsi = cls._compute_rsi(returns)

        return RegimeFeatures(
            volatility=volatility,
            trend_strength=trend_strength,
            momentum=momentum,
            relative_volume=relative_volume,
            price_to_poc=price_to_poc,
            price_to_vah=price_to_vah,
            price_to_val=price_to_val,
            value_area_width=value_area_width,
            rsi=rsi,
            bollinger_width=bollinger_width,
        )


class AdaptiveStrategySelector:
    """Adaptive selector that routes between RL and deterministic strategies."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        transaction_cost: float = 0.001,
        lookback_window: int = 30,
        evaluation_horizon: int = 20,
        training_step: int = 10,
        decision_interval: int = 5,
        min_holding_period: int = 10,
        confidence_threshold: float = 0.15,
        backend: str = 'auto',
        seed: Optional[int] = None,
    ) -> None:
        self.initial_capital = float(initial_capital)
        self.transaction_cost = float(transaction_cost)
        self.lookback_window = int(lookback_window)
        self.evaluation_horizon = int(evaluation_horizon)
        self.training_step = int(training_step)
        self.decision_interval = int(decision_interval)
        self.min_holding_period = int(min_holding_period)
        self.confidence_threshold = float(confidence_threshold)
        self.backend = backend
        self.seed = seed
        self.strategy_names = ['dqn', 'ppo', 'a3c', 'volume_profile', 'cash']
        self.model = None
        self.model_trained = False
        self.training_samples = 0
        self.training_label_counts: Dict[str, int] = {}
        self.active_strategy = 'cash'
        self._last_switch_step = -10**9
        self._last_features: Optional[RegimeFeatures] = None

    def build_runtime_strategies(self) -> Dict[str, object]:
        return {
            'dqn': get_agent('dqn', backend=self.backend, state_size=6, action_size=3, seed=self.seed),
            'ppo': get_agent('ppo', backend=self.backend, state_size=6, action_size=3, seed=self.seed),
            'a3c': get_agent('a3c', backend=self.backend, state_size=6, action_size=3, seed=self.seed),
            'volume_profile': VolumeProfileStrategy(),
            'cash': HoldStrategy(),
        }

    def fit(self, prices: Sequence[float], volumes: Sequence[float]) -> None:
        prices_arr = np.asarray(prices, dtype=np.float64)
        volumes_arr = np.asarray(volumes, dtype=np.float64)
        X: List[np.ndarray] = []
        y: List[str] = []

        start = self.lookback_window
        stop = prices_arr.size - self.evaluation_horizon
        for idx in range(start, max(start, stop), max(1, self.training_step)):
            history_prices = prices_arr[max(0, idx - self.lookback_window):idx]
            history_volumes = volumes_arr[max(0, idx - self.lookback_window):idx]
            eval_prices = prices_arr[idx - self.lookback_window: idx + self.evaluation_horizon]
            eval_volumes = volumes_arr[idx - self.lookback_window: idx + self.evaluation_horizon]
            if eval_prices.size < max(self.lookback_window + 2, self.evaluation_horizon):
                continue

            feature_row = MarketRegimeFeatureExtractor.extract(history_prices, history_volumes, self.lookback_window)
            scores = self._evaluate_candidate_scores(eval_prices, eval_volumes)
            label = max(scores.items(), key=lambda item: item[1])[0]
            X.append(feature_row.as_array())
            y.append(label)

        self.training_samples = len(X)
        self.training_label_counts = dict(Counter(y))

        if RandomForestClassifier is not None and len(X) >= 3 and len(set(y)) >= 2:
            self.model = RandomForestClassifier(
                n_estimators=64,
                max_depth=6,
                random_state=self.seed,
                class_weight='balanced_subsample',
            )
            self.model.fit(np.asarray(X, dtype=np.float32), np.asarray(y))
            self.model_trained = True
        else:
            self.model = None
            self.model_trained = False

    def _make_env(self, prices: np.ndarray, volumes: np.ndarray) -> TradingEnvironmentWithVolumeProfile:
        env = TradingEnvironmentWithVolumeProfile(
            stock_symbol='SIM',
            initial_balance=self.initial_capital,
            transaction_cost=self.transaction_cost,
            lookback_days=min(self.lookback_window, max(2, len(prices) - 1)),
            seed=self.seed,
        )
        env.prices = np.asarray(prices, dtype=np.float64)
        env.volumes = np.asarray(volumes, dtype=np.float64)
        return env

    def _simulate_strategy(self, strategy_name: str, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, float]:
        env = self._make_env(prices, volumes)
        state = env.reset()
        if strategy_name in ('dqn', 'ppo', 'a3c'):
            strategy = get_agent(strategy_name, backend=self.backend, state_size=6, action_size=3, seed=self.seed)
        elif strategy_name == 'volume_profile':
            strategy = VolumeProfileStrategy()
        else:
            strategy = HoldStrategy()

        done = False
        while not done:
            if strategy_name in ('dqn', 'ppo', 'a3c'):
                action = strategy.act(state)
            else:
                action = strategy.act(env, state)

            next_state, reward, done = env.step(action)

            if isinstance(strategy, DQNAgent):
                strategy.remember(state, action, reward, next_state, done)
                strategy.train(batch_size=min(32, max(1, len(strategy.memory))))
                if done:
                    strategy.decay_epsilon()
            elif isinstance(strategy, PPOAgent):
                value = 0.0
                if getattr(strategy, 'critic', None) is not None:
                    try:
                        import torch as _torch
                        tensor_state = _torch.tensor(state, dtype=_torch.float32, device=strategy.device).unsqueeze(0)
                        with _torch.no_grad():
                            value = float(strategy.critic(tensor_state).item())
                    except Exception:
                        value = 0.0
                strategy.store_transition(state, action, reward, value)
                if done:
                    strategy.train()
            elif isinstance(strategy, A3CAgent):
                strategy.store_transition(state, action, reward)
                if done:
                    strategy.train()

            state = next_state

        return self._score_completed_run(env)

    def _score_completed_run(self, env: TradingEnvironmentWithVolumeProfile) -> Dict[str, float]:
        final_portfolio = env.portfolio_history[-1] if env.portfolio_history else env.initial_balance
        total_return_pct = ((final_portfolio - env.initial_balance) / (env.initial_balance + 1e-8)) * 100.0
        sharpe = calculate_sharpe_ratio(env.returns_history)
        sortino = calculate_sortino_ratio(env.returns_history)
        max_drawdown = calculate_max_drawdown(env.portfolio_history)
        trade_metrics = calculate_trade_metrics(env.trades)

        sharpe_component = 0.0 if not np.isfinite(sharpe) else sharpe
        sortino_component = 0.0 if not np.isfinite(sortino) else min(sortino, 10.0)
        utility = (
            total_return_pct
            + 0.35 * sharpe_component
            + 0.25 * sortino_component
            + 0.02 * trade_metrics['win_rate']
            - 80.0 * abs(max_drawdown)
            - 0.1 * trade_metrics['total_trades'] * self.transaction_cost * 100.0
        )

        return {
            'final_portfolio': float(final_portfolio),
            'total_return_pct': float(total_return_pct),
            'sharpe_ratio': float(sharpe_component),
            'sortino_ratio': float(sortino_component),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(trade_metrics['win_rate']),
            'utility': float(utility),
        }

    def _evaluate_candidate_scores(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, float]:
        return {
            name: self._simulate_strategy(name, prices, volumes)['utility']
            for name in self.strategy_names
        }

    def _heuristic_scores(self, features: RegimeFeatures) -> Dict[str, float]:
        trend = features.trend_strength
        volatility = features.volatility
        momentum = features.momentum
        inside_value = 1.0 if features.price_to_val >= 0.0 and features.price_to_vah <= 0.0 else 0.0
        near_val = 1.0 - min(abs(features.price_to_val) * 10.0, 1.0)
        near_vah = 1.0 - min(abs(features.price_to_vah) * 10.0, 1.0)
        rsi_centered = abs((features.rsi - 50.0) / 50.0)

        return {
            'dqn': (1.2 * inside_value) + (0.8 * near_val) + (0.7 * (1.0 - min(abs(trend) * 10.0, 1.0))) - (0.6 * volatility),
            'ppo': (1.4 * max(trend, 0.0)) + (0.8 * max(momentum, 0.0)) + (0.3 * features.relative_volume) - (0.4 * volatility),
            'a3c': (1.2 * volatility) + (0.6 * abs(momentum)) + (0.5 * features.relative_volume) + (0.3 * features.bollinger_width),
            'volume_profile': (1.3 * inside_value) + (0.9 * near_val) + (0.9 * near_vah) - (0.4 * abs(trend) * 10.0) - (0.2 * rsi_centered),
            'cash': (1.4 * max(volatility - 0.03, 0.0)) + (0.7 * max(rsi_centered - 0.6, 0.0)) - (0.5 * max(trend, 0.0)),
        }

    def _score_with_model(self, features: RegimeFeatures) -> Tuple[str, Dict[str, float]]:
        if self.model_trained and self.model is not None:
            probs = self.model.predict_proba(features.as_array().reshape(1, -1))[0]
            score_map = {
                cls_name: float(prob)
                for cls_name, prob in zip(self.model.classes_, probs)
            }
            for strategy_name in self.strategy_names:
                score_map.setdefault(strategy_name, 0.0)
            selected = max(score_map.items(), key=lambda item: item[1])[0]
            return selected, score_map

        score_map = self._heuristic_scores(features)
        selected = max(score_map.items(), key=lambda item: item[1])[0]
        finite_scores = np.array(list(score_map.values()), dtype=np.float64)
        finite_scores -= np.max(finite_scores)
        probs = np.exp(finite_scores)
        probs /= np.sum(probs) + 1e-8
        return selected, {name: float(prob) for name, prob in zip(score_map.keys(), probs)}

    def select_strategy(self, env: TradingEnvironmentWithVolumeProfile) -> Tuple[str, float, Dict[str, float]]:
        history_prices = env.prices[max(0, env.current_step - self.lookback_window): env.current_step + 1]
        history_volumes = env.volumes[max(0, env.current_step - self.lookback_window): env.current_step + 1]
        features = MarketRegimeFeatureExtractor.extract(history_prices, history_volumes, self.lookback_window)
        self._last_features = features

        if (env.current_step - self._last_switch_step) < self.min_holding_period and self.active_strategy in self.strategy_names:
            return self.active_strategy, 1.0, {self.active_strategy: 1.0}

        if env.current_step % max(1, self.decision_interval) != 0 and self.active_strategy in self.strategy_names:
            return self.active_strategy, 1.0, {self.active_strategy: 1.0}

        candidate, scores = self._score_with_model(features)
        ordered = sorted(scores.values(), reverse=True)
        top_score = ordered[0] if ordered else 0.0
        runner_up = ordered[1] if len(ordered) > 1 else 0.0
        confidence = float(top_score - runner_up)

        if candidate != self.active_strategy and confidence < self.confidence_threshold and self.active_strategy in self.strategy_names:
            return self.active_strategy, confidence, scores

        if candidate != self.active_strategy:
            self.active_strategy = candidate
            self._last_switch_step = env.current_step

        return self.active_strategy, confidence, scores

    def act(self, env: TradingEnvironmentWithVolumeProfile, state: np.ndarray, runtime_strategies: Dict[str, object]) -> Tuple[int, str, float, Dict[str, float]]:
        strategy_name, confidence, scores = self.select_strategy(env)
        strategy = runtime_strategies[strategy_name]
        if strategy_name in ('dqn', 'ppo', 'a3c'):
            action = strategy.act(state)
        else:
            action = strategy.act(env, state)
        return action, strategy_name, confidence, scores

    @property
    def last_features(self) -> Optional[RegimeFeatures]:
        return self._last_features
