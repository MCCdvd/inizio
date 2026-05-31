"""
Backtesting framework for trading strategies
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from trading_agent import TradingEnvironmentWithVolumeProfile
from agents import DQNAgent, PPOAgent, A3CAgent
from visualization import VolumeProfileVisualizer
import argparse


class BacktestEngine:
    """Backtesting engine for trading strategies"""
    
    def __init__(self, stock_symbol: str, initial_capital: float = 10000, transaction_cost: float = 0.001):
        self.stock_symbol = stock_symbol
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
    
    def run_backtest(self, start_date: str, end_date: str, algorithm: str = 'dqn', episodes: int = 50):
        """Run backtest with specified algorithm"""
        print(f"\nBacktesting {self.stock_symbol} from {start_date} to {end_date}")
        print(f"Algorithm: {algorithm.upper()}, Episodes: {episodes}\n")
        
        env = TradingEnvironmentWithVolumeProfile(self.stock_symbol, initial_balance=self.initial_capital)
        
        env.load_data(start_date, end_date)
        
        if algorithm.lower() == 'dqn':
            agent = DQNAgent()
        elif algorithm.lower() == 'ppo':
            agent = PPOAgent()
        elif algorithm.lower() == 'a3c':
            agent = A3CAgent()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        episode_returns = []
        best_portfolio = self.initial_capital
        best_episode = 0
        
        for episode in range(episodes):
            state = env.reset()
            
            while True:
                if algorithm.lower() == 'dqn':
                    action = agent.act(state)
                    next_state, reward, done = env.step(action)
                    agent.remember(state, action, reward, next_state, done)
                    
                    if done:
                        break
                    agent.train(batch_size=32)
                    agent.decay_epsilon()
                    
                elif algorithm.lower() == 'ppo':
                    action = agent.act(state)
                    if agent.critic:
                        value = agent.critic.predict(state.reshape(1, -1), verbose=0)[0][0]
                    else:
                        value = 0
                    next_state, reward, done = env.step(action)
                    agent.store_transition(state, action, reward, value)
                    
                    if done:
                        agent.train()
                        break
                
                elif algorithm.lower() == 'a3c':
                    action = agent.act(state)
                    next_state, reward, done = env.step(action)
                    agent.store_transition(state, action, reward)
                    
                    if done:
                        agent.train()
                        break
                
                state = next_state
            
            portfolio_value = env.balance + (env.shares_held * env.prices[env.current_step - 1])
            returns = ((portfolio_value - self.initial_capital) / self.initial_capital) * 100
            episode_returns.append(returns)
            
            if portfolio_value > best_portfolio:
                best_portfolio = portfolio_value
                best_episode = episode + 1
            
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode+1}: Portfolio ${portfolio_value:,.2f}, Return {returns:+.2f}%")
        
        # Calculate metrics
        final_portfolio = env.balance + (env.shares_held * env.prices[-1])
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
        self._plot_results(env, results)
        
        return results
    
    def _print_results(self, results):
        """Print backtest results"""
        print(f"\n{'='*60}")
        print(f"BACKTEST RESULTS - {results['symbol']}")
        print(f"{'='*60}\n")
        print(f"Period: {results['start_date']} to {results['end_date']}")
        print(f"Algorithm: {results['algorithm'].upper()}")
        print(f"\nCapital:")
        print(f"  Initial: ${results['initial_capital']:,.2f}")
        print(f"  Final: ${results['final_portfolio']:,.2f}")
        print(f"  Return: {results['total_return_pct']:+.2f}%")
        print(f"\nTrading:")
        print(f"  Total Trades: {results['total_trades']}")
        print(f"  Buy Trades: {results['buy_trades']}")
        print(f"  Sell Trades: {results['sell_trades']}")
        print(f"  Winning Trades: {results['winning_trades']}")
        print(f"  Win Rate: {results['win_rate_pct']:.1f}%")
        print(f"\nPerformance:")
        print(f"  Avg Return/Trade: {results['avg_return_per_trade']:+.2f}%")
        print(f"  Best Episode: #{results['best_episode']}")
        print(f"  Best Portfolio Value: ${results['best_portfolio']:,.2f}")
        print(f"\n{'='*60}\n")
    
    def _plot_results(self, env, results):
        """Plot backtest results"""
        VolumeProfileVisualizer.plot_volume_profile(
            env.prices, env.volumes, env.poc, env.vah, env.val,
            trades=results['trades'],
            title=f"{results['algorithm'].upper()} Backtest - {results['symbol']}"
        )
        VolumeProfileVisualizer.plot_training_results(
            [0] * len(results['episode_returns']),
            [10000 * (1 + r/100) for r in results['episode_returns']],
            title=f"{results['algorithm'].upper()} Backtest - {results['symbol']}"
        )
        
        import matplotlib.pyplot as plt
        plt.show()


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
    
    args = parser.parse_args()
    
    engine = BacktestEngine(args.stock, initial_capital=args.capital)
    results = engine.run_backtest(
        start_date=args.start_date,
        end_date=args.end_date,
        algorithm=args.algorithm,
        episodes=args.episodes
    )
