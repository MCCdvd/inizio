import argparse
import csv
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from trading_agent import TradingEnvironmentWithVolumeProfile
from agents import DQNAgent, PPOAgent, A3CAgent
from visualization import VolumeProfileVisualizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BacktestEngine:
    """Backtesting engine for trading strategies"""

    def __init__(
        self,
        stock_symbol: str,
        initial_capital: float = 10000,
        transaction_cost: float = 0.001,
        seed: int = None,
    ):
        self.stock_symbol = stock_symbol
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def run_backtest(
        self,
        start_date: str,
        end_date: str,
        algorithm: str = "dqn",
        episodes: int = 50,
        backend: str = "auto",
        output_csv: str = None,
        output_json: str = None,
        output_dir: str = None,
        no_plot: bool = False,
    ):
        """Run backtest with specified algorithm"""
        logger.info(f"Backtesting {self.stock_symbol} from {start_date} to {end_date}")
        logger.info(f"Algorithm: {algorithm.upper()}, Episodes: {episodes}, Backend: {backend}")

        # Sprint 2: output dir handling
        out_dir = None
        if output_dir:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            if output_csv is None:
                output_csv = str(out_dir / "metrics.csv")
            if output_json is None:
                output_json = str(out_dir / "summary.json")

        env = TradingEnvironmentWithVolumeProfile(
            self.stock_symbol,
            initial_balance=self.initial_capital,
            seed=self.seed,
        )
        env.load_data(start_date, end_date)

        # select agent
        if algorithm.lower() == "dqn":
            agent = DQNAgent(seed=self.seed)
        elif algorithm.lower() == "ppo":
            agent = PPOAgent(seed=self.seed)
        elif algorithm.lower() == "a3c":
            agent = A3CAgent(seed=self.seed)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        episode_returns = []
        best_portfolio = self.initial_capital
        best_episode = 0

        for episode in range(episodes):
            state = env.reset()

            while True:
                action = agent.act(state)
                next_state, reward, done = env.step(action)

                # training bookkeeping
                if isinstance(agent, DQNAgent):
                    agent.remember(state, action, reward, next_state, done)
                    agent.train(batch_size=32)
                    if done:
                        agent.decay_epsilon()

                elif isinstance(agent, PPOAgent):
                    if hasattr(agent, "critic") and agent.critic is not None:
                        try:
                            import torch as _torch

                            _t = _torch.tensor(state, dtype=_torch.float32).unsqueeze(0)
                            with _torch.no_grad():
                                value = float(agent.critic(_t).item())
                        except Exception:
                            try:
                                value = float(
                                    agent.critic.predict(state.reshape(1, -1), verbose=0)[0][0]
                                )
                            except Exception:
                                value = 0.0
                    else:
                        value = 0.0

                    agent.store_transition(state, action, reward, value)
                    if done:
                        agent.train()

                elif isinstance(agent, A3CAgent):
                    agent.store_transition(state, action, reward)
                    if done:
                        agent.train()

                state = next_state
                if done:
                    break

            portfolio_value = env.balance + (
                env.shares_held * env.prices[min(env.current_step - 1, len(env.prices) - 1)]
            )
            returns = ((portfolio_value - self.initial_capital) / self.initial_capital) * 100
            episode_returns.append(float(returns))

            if portfolio_value > best_portfolio:
                best_portfolio = portfolio_value
                best_episode = episode + 1

            if (episode + 1) % 10 == 0:
                logger.info(
                    f"Episode {episode+1}: Portfolio ${portfolio_value:,.2f}, Return {returns:+.2f}%"
                )

        # Calculate metrics
        final_portfolio = (
            env.balance + (env.shares_held * env.prices[-1]) if len(env.prices) > 0 else env.balance
        )
        final_return = ((final_portfolio - self.initial_capital) / self.initial_capital) * 100

        buy_trades = [t for t in env.trades if t["type"] == "BUY"]
        sell_trades = [t for t in env.trades if t["type"] == "SELL"]

        winning_trades = len([t for t in sell_trades if t.get("profit_pct", 0) > 0])
        win_rate = (winning_trades / len(sell_trades) * 100) if len(sell_trades) > 0 else 0
        avg_return_per_trade = final_return / len(env.trades) if len(env.trades) > 0 else 0

        results = {
            "symbol": self.stock_symbol,
            "start_date": start_date,
            "end_date": end_date,
            "algorithm": algorithm,
            "initial_capital": float(self.initial_capital),
            "final_portfolio": float(final_portfolio),
            "total_return_pct": float(final_return),
            "total_trades": int(len(env.trades)),
            "buy_trades": int(len(buy_trades)),
            "sell_trades": int(len(sell_trades)),
            "winning_trades": int(winning_trades),
            "win_rate_pct": float(win_rate),
            "avg_return_per_trade": float(avg_return_per_trade),
            "best_episode": int(best_episode),
            "best_portfolio_value": float(best_portfolio),
            "episode_returns": [float(x) for x in episode_returns],
            "trades": env.trades,
        }

        logger.info("=" * 60)
        logger.info(f"BACKTEST RESULTS - {self.stock_symbol}")
        logger.info("=" * 60)
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Algorithm: {algorithm.upper()}")
        logger.info("")
        logger.info("Capital:")
        logger.info(f"  Initial: ${self.initial_capital:,.2f}")
        logger.info(f"  Final: ${final_portfolio:,.2f}")
        logger.info(f"  Return: {final_return:+.2f}%")
        logger.info("")
        logger.info("Trading:")
        logger.info(f"  Total Trades: {len(env.trades)}")
        logger.info(f"  Buy Trades: {len(buy_trades)}")
        logger.info(f"  Sell Trades: {len(sell_trades)}")
        logger.info(f"  Winning Trades: {winning_trades}")
        logger.info(f"  Win Rate: {win_rate:.1f}%")
        logger.info("")
        logger.info("Performance:")
        logger.info(f"  Avg Return/Trade: {avg_return_per_trade:+.2f}%")
        logger.info(f"  Best Episode: #{best_episode}")
        logger.info(f"  Best Portfolio Value: ${best_portfolio:,.2f}")
        logger.info("=" * 60)

        # Sprint 2: save metrics.csv
        if output_csv:
            try:
                with open(output_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["episode", "return_pct", "portfolio_value"])
                    for i, r in enumerate(episode_returns, start=1):
                        pv = self.initial_capital * (1 + (float(r) / 100.0))
                        writer.writerow([i, float(r), float(pv)])
                logger.info(f"Saved metrics CSV to {output_csv}")
            except Exception as e:
                logger.warning(f"Failed to save CSV '{output_csv}': {e}")

        # Sprint 2: save summary.json
        if output_json:
            try:
                with open(output_json, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, default=float)
                logger.info(f"Saved summary JSON to {output_json}")
            except Exception as e:
                logger.warning(f"Failed to save JSON '{output_json}': {e}")

        # Sprint 2: plot handling
        if not no_plot:
            self._plot_results(env, results, output_dir=output_dir)

        return results

    def _plot_results(self, env, results, output_dir: str = None):
        """Plot backtest results"""
        try:
            VolumeProfileVisualizer.plot_volume_profile(
                env.prices,
                env.volumes,
                env.poc,
                env.vah,
                env.val,
                trades=results["trades"],
                title=f"{results['algorithm'].upper()} Backtest - {results['symbol']}",
            )
            VolumeProfileVisualizer.plot_training_results(
                [0] * len(results["episode_returns"]),
                [self.initial_capital * (1 + r / 100) for r in results["episode_returns"]],
                title=f"{results['algorithm'].upper()} Backtest - {results['symbol']}",
                baseline_capital=self.initial_capital,
            )

            import matplotlib.pyplot as plt

            if output_dir:
                out_dir = Path(output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                fig = plt.gcf()
                fig.savefig(out_dir / "equity_curve.png", dpi=150, bbox_inches="tight")
                logger.info(f"Saved equity curve to {out_dir / 'equity_curve.png'}")
            else:
                plt.show()

        except Exception as e:
            logger.warning(f"Plotting failed (results are still valid): {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Backtest trading agent strategies")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Stock symbol")
    parser.add_argument(
        "--start-date",
        type=str,
        default=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="dqn",
        choices=["dqn", "ppo", "a3c"],
        help="RL algorithm",
    )
    parser.add_argument("--episodes", type=int, default=50, help="Training episodes")
    parser.add_argument("--initial-capital", type=float, default=10000.0, help="Initial capital")
    parser.add_argument("--transaction-cost", type=float, default=0.001, help="Transaction cost")
    parser.add_argument("--backend", type=str, default="auto", help="Data backend")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional CSV output path")
    parser.add_argument("--output-json", type=str, default=None, help="Optional JSON output path")
    parser.add_argument("--output-dir", type=str, default="output", help="Artifacts directory")
    parser.add_argument("--no-plot", action="store_true", help="Disable plots")
    return parser.parse_args()


if __name__ == "__main__":
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