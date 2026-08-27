"""Backtesting engine for fixed and adaptive trading strategies."""

import argparse
import csv
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

from agents import A3CAgent, DQNAgent, PPOAgent, get_agent
from strategy_selector import AdaptiveStrategySelector
from trading_agent import TradingEnvironmentWithVolumeProfile
from utils import (
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_trade_metrics,
)
from visualization import VolumeProfileVisualizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BacktestEngine:
    """Backtesting engine for trading strategies."""

    def __init__(
        self,
        stock_symbol: str,
        initial_capital: float = 10000,
        transaction_cost: float = 0.001,
        seed: Optional[int] = None,
    ):
        self.stock_symbol = stock_symbol
        self.initial_capital = float(initial_capital)
        self.transaction_cost = float(transaction_cost)
        self.seed = seed

    def _create_environment(self, start_date: str, end_date: str) -> TradingEnvironmentWithVolumeProfile:
        env = TradingEnvironmentWithVolumeProfile(
            self.stock_symbol,
            initial_balance=self.initial_capital,
            transaction_cost=self.transaction_cost,
            seed=self.seed,
        )
        env.load_data(start_date, end_date)
        return env

    def _train_selected_agent(self, agent, state, action, reward, next_state, done) -> None:
        if isinstance(agent, DQNAgent):
            agent.remember(state, action, reward, next_state, done)
            agent.train(batch_size=min(32, max(1, len(agent.memory))))
            if done:
                agent.decay_epsilon()
        elif isinstance(agent, PPOAgent):
            value = 0.0
            if getattr(agent, 'critic', None) is not None:
                try:
                    import torch as _torch

                    tensor_state = _torch.tensor(state, dtype=_torch.float32, device=agent.device).unsqueeze(0)
                    with _torch.no_grad():
                        value = float(agent.critic(tensor_state).item())
                except Exception:
                    value = 0.0
            agent.store_transition(state, action, reward, value)
            if done:
                agent.train()
        elif isinstance(agent, A3CAgent):
            agent.store_transition(state, action, reward)
            if done:
                agent.train()

    def _finalize_results(
        self,
        env: TradingEnvironmentWithVolumeProfile,
        algorithm: str,
        episode_returns,
        best_episode: int,
        best_portfolio: float,
        selector_details: Optional[Dict] = None,
    ) -> Dict:
        final_portfolio = env.portfolio_history[-1] if env.portfolio_history else env.initial_balance
        final_return = ((final_portfolio - self.initial_capital) / (self.initial_capital + 1e-8)) * 100.0
        buy_trades = [t for t in env.trades if t['type'] == 'BUY']
        sell_trades = [t for t in env.trades if t['type'] == 'SELL']
        winning_trades = len([t for t in sell_trades if t.get('profit_pct', 0) > 0])
        win_rate = (winning_trades / len(sell_trades) * 100.0) if sell_trades else 0.0
        avg_return_per_trade = final_return / len(env.trades) if env.trades else 0.0
        sharpe_ratio = calculate_sharpe_ratio(env.returns_history)
        sortino_ratio = calculate_sortino_ratio(env.returns_history)
        max_drawdown = calculate_max_drawdown(env.portfolio_history)
        trade_metrics = calculate_trade_metrics(env.trades)
        total_fees = sum(float(t.get('fee', 0.0)) for t in env.trades)

        results = {
            'symbol': self.stock_symbol,
            'algorithm': algorithm,
            'initial_capital': float(self.initial_capital),
            'final_portfolio': float(final_portfolio),
            'total_return_pct': float(final_return),
            'total_trades': int(len(env.trades)),
            'buy_trades': int(len(buy_trades)),
            'sell_trades': int(len(sell_trades)),
            'winning_trades': int(winning_trades),
            'win_rate_pct': float(win_rate),
            'avg_return_per_trade': float(avg_return_per_trade),
            'best_episode': int(best_episode),
            'best_portfolio_value': float(best_portfolio),
            'episode_returns': [float(x) for x in episode_returns],
            'trades': env.trades,
            'portfolio_history': [float(x) for x in env.portfolio_history],
            'period_returns': [float(x) for x in env.returns_history],
            'sharpe_ratio': float(sharpe_ratio) if sharpe_ratio == sharpe_ratio else 0.0,
            'sortino_ratio': float(sortino_ratio) if sortino_ratio == sortino_ratio else 0.0,
            'max_drawdown': float(max_drawdown),
            'trade_metrics': trade_metrics,
            'transaction_cost': float(self.transaction_cost),
            'total_fees_paid': float(total_fees),
        }
        if selector_details:
            results['selector'] = selector_details
        return results

    def _run_single_algorithm(self, env: TradingEnvironmentWithVolumeProfile, algorithm: str, episodes: int, backend: str):
        agent = get_agent(algorithm, backend=backend, state_size=6, action_size=3, seed=self.seed)
        episode_returns = []
        best_portfolio = self.initial_capital
        best_episode = 0

        for episode in range(episodes):
            state = env.reset()
            done = False
            while not done:
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                self._train_selected_agent(agent, state, action, reward, next_state, done)
                state = next_state

            portfolio_value = env.portfolio_history[-1] if env.portfolio_history else env.initial_balance
            returns = ((portfolio_value - self.initial_capital) / (self.initial_capital + 1e-8)) * 100.0
            episode_returns.append(float(returns))
            if portfolio_value > best_portfolio:
                best_portfolio = portfolio_value
                best_episode = episode + 1
            if (episode + 1) % 10 == 0:
                logger.info(
                    'Episode %d: Portfolio $%0.2f, Return %+0.2f%%',
                    episode + 1,
                    portfolio_value,
                    returns,
                )

        return self._finalize_results(env, algorithm, episode_returns, best_episode, best_portfolio)

    def _run_adaptive(self, env: TradingEnvironmentWithVolumeProfile, episodes: int, backend: str) -> Dict:
        selector = AdaptiveStrategySelector(
            initial_capital=self.initial_capital,
            transaction_cost=self.transaction_cost,
            backend=backend,
            seed=self.seed,
        )
        selector.fit(env.prices, env.volumes)
        runtime_strategies = selector.build_runtime_strategies()
        episode_returns = []
        best_portfolio = self.initial_capital
        best_episode = 0
        strategy_usage: Dict[str, int] = {name: 0 for name in selector.strategy_names}
        strategy_switches = []
        confidence_history = []
        feature_snapshots = []

        for episode in range(episodes):
            state = env.reset()
            done = False
            previous_strategy = selector.active_strategy
            while not done:
                action, strategy_name, confidence, scores = selector.act(env, state, runtime_strategies)
                next_state, reward, done = env.step(action)
                self._train_selected_agent(runtime_strategies[strategy_name], state, action, reward, next_state, done)
                strategy_usage[strategy_name] += 1
                confidence_history.append(float(confidence))

                if strategy_name != previous_strategy:
                    strategy_switches.append({
                        'episode': int(episode + 1),
                        'step': int(env.current_step),
                        'strategy': strategy_name,
                        'confidence': float(confidence),
                        'scores': {k: float(v) for k, v in scores.items()},
                    })
                    previous_strategy = strategy_name

                if selector.last_features is not None and len(feature_snapshots) < 25:
                    feature_snapshots.append({
                        'step': int(env.current_step),
                        'strategy': strategy_name,
                        **selector.last_features.as_dict(),
                    })

                state = next_state

            portfolio_value = env.portfolio_history[-1] if env.portfolio_history else env.initial_balance
            returns = ((portfolio_value - self.initial_capital) / (self.initial_capital + 1e-8)) * 100.0
            episode_returns.append(float(returns))
            if portfolio_value > best_portfolio:
                best_portfolio = portfolio_value
                best_episode = episode + 1
            if (episode + 1) % 5 == 0 or episode == episodes - 1:
                logger.info(
                    'Adaptive episode %d: Portfolio $%0.2f, Return %+0.2f%%, Strategy %s',
                    episode + 1,
                    portfolio_value,
                    returns,
                    selector.active_strategy,
                )

        selector_details = {
            'model_trained': selector.model_trained,
            'training_samples': int(selector.training_samples),
            'training_label_counts': selector.training_label_counts,
            'active_strategy': selector.active_strategy,
            'decision_interval': selector.decision_interval,
            'min_holding_period': selector.min_holding_period,
            'confidence_threshold': selector.confidence_threshold,
            'strategy_usage': strategy_usage,
            'strategy_switches': strategy_switches,
            'confidence_history': confidence_history,
            'feature_snapshots': feature_snapshots,
        }

        return self._finalize_results(env, 'adaptive', episode_returns, best_episode, best_portfolio, selector_details)

    def run_backtest(
        self,
        start_date: str,
        end_date: str,
        algorithm: str = 'dqn',
        episodes: int = 50,
        backend: str = 'auto',
        output_csv: str = None,
        output_json: str = None,
        output_dir: str = None,
        no_plot: bool = False,
    ) -> Dict:
        logger.info('Backtesting %s from %s to %s', self.stock_symbol, start_date, end_date)
        logger.info('Algorithm: %s, Episodes: %d, Backend: %s', algorithm.upper(), episodes, backend)

        if episodes <= 0:
            raise ValueError('--episodes must be > 0')
        if start_date > end_date:
            raise ValueError('--start-date must be <= --end-date')

        out_dir = None
        if output_dir:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            if output_csv is None:
                output_csv = str(out_dir / 'metrics.csv')
            if output_json is None:
                output_json = str(out_dir / 'summary.json')

        env = self._create_environment(start_date, end_date)
        if algorithm.lower() == 'adaptive':
            results = self._run_adaptive(env, episodes, backend)
        elif algorithm.lower() in {'dqn', 'ppo', 'a3c'}:
            results = self._run_single_algorithm(env, algorithm.lower(), episodes, backend)
        else:
            raise ValueError(f'Unknown algorithm: {algorithm}')

        logger.info('=' * 60)
        logger.info('BACKTEST RESULTS - %s', self.stock_symbol)
        logger.info('=' * 60)
        logger.info('Algorithm: %s', results['algorithm'].upper())
        logger.info('Initial: $%0.2f', self.initial_capital)
        logger.info('Final:   $%0.2f', results['final_portfolio'])
        logger.info('Return:  %+0.2f%%', results['total_return_pct'])
        logger.info('Sharpe:  %0.3f', results['sharpe_ratio'])
        logger.info('Sortino: %0.3f', results['sortino_ratio'])
        logger.info('Max DD:  %0.3f', results['max_drawdown'])
        logger.info('Trades:  %d', results['total_trades'])
        logger.info('Fees:    $%0.2f', results['total_fees_paid'])
        if 'selector' in results:
            logger.info('Selector trained: %s', results['selector']['model_trained'])
            logger.info('Selector usage: %s', results['selector']['strategy_usage'])
        logger.info('=' * 60)

        if output_csv:
            with open(output_csv, 'w', newline='', encoding='utf-8') as handle:
                writer = csv.writer(handle)
                writer.writerow(['step', 'portfolio_value', 'period_return'])
                for idx, portfolio_value in enumerate(results['portfolio_history']):
                    period_return = results['period_returns'][idx - 1] if idx > 0 and idx - 1 < len(results['period_returns']) else 0.0
                    writer.writerow([idx, float(portfolio_value), float(period_return)])
            logger.info('Saved metrics CSV to %s', output_csv)

        if output_json:
            with open(output_json, 'w', encoding='utf-8') as handle:
                json.dump(results, handle, indent=2)
            logger.info('Saved summary JSON to %s', output_json)

        if not no_plot:
            self._plot_results(env, results, output_dir=output_dir)

        return results

    def _plot_results(self, env, results, output_dir: str = None):
        try:
            VolumeProfileVisualizer.plot_volume_profile(
                env.prices,
                env.volumes,
                env.poc,
                env.vah,
                env.val,
                trades=results['trades'],
                title=f"{results['algorithm'].upper()} Backtest - {results['symbol']}",
            )
            VolumeProfileVisualizer.plot_training_results(
                [0.0] + results['period_returns'],
                results['portfolio_history'],
                title=f"{results['algorithm'].upper()} Backtest - {results['symbol']}",
            )

            import matplotlib.pyplot as plt

            if output_dir:
                out_dir = Path(output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                plt.gcf().savefig(out_dir / 'equity_curve.png', dpi=150, bbox_inches='tight')
                logger.info('Saved equity curve to %s', out_dir / 'equity_curve.png')
            else:
                plt.show()
        except Exception as exc:  # pragma: no cover - plotting failures are non-fatal
            logger.warning('Plotting failed (results are still valid): %s', exc)


def parse_args():
    parser = argparse.ArgumentParser(description='Backtest trading agent strategies')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol')
    parser.add_argument(
        '--start-date',
        type=str,
        default=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
        help='Start date (YYYY-MM-DD)',
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=datetime.now(UTC).strftime('%Y-%m-%d'),
        help='End date (YYYY-MM-DD)',
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        default='dqn',
        choices=['dqn', 'ppo', 'a3c', 'adaptive'],
        help='Trading strategy to backtest',
    )
    parser.add_argument('--episodes', type=int, default=50, help='Training episodes')
    parser.add_argument('--initial-capital', type=float, default=10000.0, help='Initial capital')
    parser.add_argument('--transaction-cost', type=float, default=0.001, help='Transaction cost')
    parser.add_argument('--backend', type=str, default='auto', help='Model backend')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--output-csv', type=str, default=None, help='Optional CSV output path')
    parser.add_argument('--output-json', type=str, default=None, help='Optional JSON output path')
    parser.add_argument('--output-dir', type=str, default='output', help='Artifacts directory')
    parser.add_argument('--no-plot', action='store_true', help='Disable plots')
    return parser.parse_args()


if __name__ == '__main__':
    try:
        args = parse_args()
        engine = BacktestEngine(
            stock_symbol=args.symbol,
            initial_capital=args.initial_capital,
            transaction_cost=args.transaction_cost,
            seed=args.seed,
        )
        engine.run_backtest(
            start_date=args.start_date,
            end_date=args.end_date,
            algorithm=args.algorithm,
            episodes=args.episodes,
            backend=args.backend,
            output_csv=args.output_csv,
            output_json=args.output_json,
            output_dir=args.output_dir,
            no_plot=args.no_plot,
        )
    except Exception as exc:
        logger.exception('Backtest failed: %s', exc)
        raise SystemExit(1)
