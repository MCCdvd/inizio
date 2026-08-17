import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from trading_agent import TradingEnvironmentWithVolumeProfile
from agents import DQNAgent, PPOAgent, A3CAgent
from visualization import VolumeProfileVisualizer
import argparse
import logging
import json

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Backtesting engine for trading strategies"""
    
    def __init__(self, stock_symbol: str, initial_capital: float = 10000, transaction_cost: float = 0.001, seed: int = None):
        self.stock_symbol = stock_symbol
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def run_backtest(self, start_date: str, end_date: str, algorithm: str = 'dqn', episodes: int = 50, backend: str = 'auto', output_csv: str = None, output_json: str = None, no_plot: bool = False):
        """Run backtest with specified algorithm"""
        logger.info(f"Backtesting {self.stock_symbol} from {start_date} to {end_date}")
        logger.info(f"Algorithm: {algorithm.upper()}, Episodes: {episodes}, Backend: {backend}")
        
        env = TradingEnvironmentWithVolumeProfile(self.stock_symbol, initial_balance=self.initial_capital, seed=self.seed)
        env.load_data(start_date, end_date)
        
        # select agent
        if algorithm.lower() == 'dqn':
            agent = DQNAgent(seed=self.seed)
        elif algorithm.lower() == 'ppo':
            agent = PPOAgent(seed=self.seed)
        elif algorithm.lower() == 'a3c':
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
                    if done:
                        agent.train(batch_size=32)
                        agent.decay_epsilon()
                    else:
                        agent.train(batch_size=32)
                elif isinstance(agent, PPOAgent):
                    if hasattr(agent, 'critic') and agent.critic is not None:
                        try:
                            import torch as _torch
                            _t = _torch.tensor(state, dtype=_torch.float32).unsqueeze(0)
                            with _torch.no_grad():
                                value = float(agent.critic(_t).item())
                        except Exception:
                            try:
                                value = float(agent.critic.predict(state.reshape(1, -1), verbose=0)[0][0])
                            except Exception:
                                value = 0
                    else:
                        value = 0
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

            portfolio_value = env.balance + (env.shares_held * env.prices[min(env.current_step - 1, len(env.prices)-1)])
            returns = ((portfolio_value - self.initial_capital) / self.initial_capital) * 100
            episode_returns.append(returns)
            
            if portfolio_value > best_portfolio:
                best_portfolio = portfolio_value
                best_episode = episode + 1
            
            if (episode + 1) % 10 == 0:
                logger.info(f"Episode {episode+1}: Portfolio ${portfolio_value:,.2f}, Return {returns:+.2f}%")
        
        # Calculate metrics
        final_portfolio = env.balance + (env.shares_held * env.prices[-1]) if len(env.prices) > 0 else env.balance
        final_return = ((final_portfolio - self.initial_capital) / self.initial_capital) * 100
        
        buy_trades = [t for t in env.trades if t['type'] == 'BUY']
        sell_trades = [t for t in env.trades if t['type'] == 'SELL']
        
        winning_trades = len([t for t in sell_trades if t.get('profit_pct', 0) > 0])
        win_rate = (winning_trades / len(sell_trades) * 100) if len(sell_trades) > 0 else 0
        
        avg_return_per_trade = final_return / len(env.trades) if len(env.trades) > 0 else 0
        
        results = {
            'symbol': self.stock_symbol,
            'start_date': start_date,
            'end_date': end_date,
            'algorithm': algorithm,
            'initial_capital': self.initial_capital,
            'final_portfolio': final_portfolio,
            'total_return_pct': final_return,
            'total_trades': len(env.trades),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'winning_trades': winning_trades,
            'win_rate_pct': win_rate,
            'avg_return_per_trade': avg_return_per_trade,
            'best_episode': best_episode,
            'best_portfolio': best_portfolio,
            'episode_returns': episode_returns,
            'trades': env.trades
        }
        
        self._print_results(results)
        if not no_plot:
            self._plot_results(env, results)

        # Export results
        if output_csv:
            try:
                df = pd.DataFrame(results['trades'])
                df.to_csv(output_csv, index=False)
                logger.info(f"Wrote trades CSV to {output_csv}")
            except Exception as e:
                logger.warning(f"Failed to write CSV: {e}")
        if output_json:
            try:
                with open(output_json, 'w') as f:
                    json.dump(results, f, default=str, indent=2)
                logger.info(f"Wrote results JSON to {output_json}")
            except Exception as e:
                logger.warning(f"Failed to write JSON: {e}")
        
        return results
    
    def _print_results(self, results):
        """Print backtest results"""
        logger.info("="*60)
        logger.info(f"BACKTEST RESULTS - {results['symbol']}")
        logger.info("="*60)
        logger.info(f"Period: {results['start_date']} to {results['end_date']}")
        logger.info(f"Algorithm: {results['algorithm'].upper()}")
        logger.info("\nCapital:")
        logger.info(f"  Initial: ${results['initial_capital']:,.2f}")
        logger.info(f"  Final: ${results['final_portfolio']:,.2f}")
        logger.info(f"  Return: {results['total_return_pct']:+.2f}%")
        logger.info("\nTrading:")
        logger.info(f"  Total Trades: {results['total_trades']}")
        logger.info(f"  Buy Trades: {results['buy_trades']}")
        logger.info(f"  Sell Trades: {results['sell_trades']}")
        logger.info(f"  Winning Trades: {results['winning_trades']}")
        logger.info(f"  Win Rate: {results['win_rate_pct']:.1f}%")
        logger.info("\nPerformance:")
        logger.info(f"  Avg Return/Trade: {results['avg_return_per_trade']:+.2f}%")
        logger.info(f"  Best Episode: #{results['best_episode']}")
        logger.info(f"  Best Portfolio Value: ${results['best_portfolio']:,.2f}")
        logger.info("="*60)
    
    def _plot_results(self, env, results):
        """Plot backtest results"""
        try:
            VolumeProfileVisualizer.plot_volume_profile(
                env.prices, env.volumes, env.poc, env.vah, env.val,
                trades=results['trades'],
                title=f"{results['algorithm'].upper()} Backtest - {results['symbol']}"
            )
            VolumeProfileVisualizer.plot_training_results(
                [0] * len(results['episode_returns']),
                [self.initial_capital * (1 + r/100) for r in results['episode_returns']],
                title=f"{results['algorithm'].upper()} Backtest - {results['symbol']}",
                baseline_capital=self.initial_capital
            )
            import matplotlib.pyplot as plt
            plt.show()
        except Exception as e:
            logger.warning(f"Plotting failed (results are still valid): {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Backtest trading strategy')
    parser.add_argument('--stock', default='AAPL', help='Stock symbol')
    parser.add_argument('--algorithm', choices=['dqn', 'ppo', 'a3c'], default='dqn', help='Algorithm')
    parser.add_argument('--start-date', default=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default=datetime.now().strftime('%Y-%m-%d'),
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--episodes', type=int, default=50, help='Number of episodes')
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--output-csv', type=str, default=None, help='Path to output trades CSV')
    parser.add_argument('--output-json', type=str, default=None, help='Path to output results JSON')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting charts')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    
    engine = BacktestEngine(args.stock, initial_capital=args.capital, seed=args.seed)
    results = engine.run_backtest(
        start_date=args.start_date,
        end_date=args.end_date,
        algorithm=args.algorithm,
        episodes=args.episodes,
        output_csv=args.output_csv,
        output_json=args.output_json,
        no_plot=args.no_plot,
    )
