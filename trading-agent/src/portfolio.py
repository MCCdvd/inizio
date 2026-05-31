"""
Multi-stock portfolio trading with RL
"""
import numpy as np
from typing import List, Dict
from datetime import datetime, timedelta
from trading_agent import TradingEnvironmentWithVolumeProfile
from agents import DQNAgent, PPOAgent, A3CAgent


class PortfolioAgent:
    """Multi-stock portfolio agent"""
    
    def __init__(self, stocks: List[str], initial_capital: float = 50000, max_position_size: float = 0.25):
        self.stocks = stocks
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.capital_per_stock = initial_capital / len(stocks)
        
        self.environments = {}
        self.agents = {}
        self.results = {}
    
    def initialize(self, algorithm: str = 'dqn'):
        """Initialize environments and agents for each stock"""
        for stock in self.stocks:
            self.environments[stock] = TradingEnvironmentWithVolumeProfile(
                stock, initial_balance=self.capital_per_stock
            )
            
            if algorithm.lower() == 'dqn':
                self.agents[stock] = DQNAgent()
            elif algorithm.lower() == 'ppo':
                self.agents[stock] = PPOAgent()
            elif algorithm.lower() == 'a3c':
                self.agents[stock] = A3CAgent()
            
            # Load data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            self.environments[stock].load_data(
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
    
    def train(self, episodes: int = 50, algorithm: str = 'dqn'):
        """Train agents on all stocks"""
        print(f"\nTraining portfolio on {len(self.stocks)} stocks")
        print(f"Algorithm: {algorithm.upper()}, Episodes: {episodes}\n")
        
        self.initialize(algorithm)
        
        for stock in self.stocks:
            print(f"Training {stock}...")
            env = self.environments[stock]
            agent = self.agents[stock]
            
            for episode in range(episodes):
                state = env.reset()
                
                while True:
                    action = agent.act(state)
                    next_state, reward, done = env.step(action)
                    
                    if algorithm.lower() == 'dqn':
                        agent.remember(state, action, reward, next_state, done)
                        if done:
                            agent.train(batch_size=32)
                            agent.decay_epsilon()
                    
                    elif algorithm.lower() == 'ppo':
                        if agent.critic:
                            value = agent.critic.predict(state.reshape(1, -1), verbose=0)[0][0]
                        else:
                            value = 0
                        agent.store_transition(state, action, reward, value)
                        if done:
                            agent.train()
                    
                    elif algorithm.lower() == 'a3c':
                        agent.store_transition(state, action, reward)
                        if done:
                            agent.train()
                    
                    state = next_state
                    if done:
                        break
                
                if (episode + 1) % 10 == 0:
                    portfolio_value = env.balance + (env.shares_held * env.prices[env.current_step - 1])
                    ret = ((portfolio_value - self.capital_per_stock) / self.capital_per_stock) * 100
                    print(f"  {stock} Episode {episode+1}: ${portfolio_value:,.2f} ({ret:+.2f}%)")
            
            # Store results
            portfolio_value = env.balance + (env.shares_held * env.prices[env.current_step - 1])
            self.results[stock] = {
                'final_portfolio': portfolio_value,
                'trades': env.trades,
                'return_pct': ((portfolio_value - self.capital_per_stock) / self.capital_per_stock) * 100
            }
        
        self._print_portfolio_summary()
    
    def _print_portfolio_summary(self):
        """Print portfolio summary"""
        print(f"\n{'='*60}")
        print(f"PORTFOLIO SUMMARY")
        print(f"{'='*60}\n")
        
        total_portfolio = sum(r['final_portfolio'] for r in self.results.values())
        total_return = ((total_portfolio - self.initial_capital) / self.initial_capital) * 100
        
        print(f"Total Capital: ${self.initial_capital:,.2f}")
        print(f"Final Portfolio: ${total_portfolio:,.2f}")
        print(f"Total Return: {total_return:+.2f}%\n")
        
        for stock in self.stocks:
            result = self.results[stock]
            print(f"{stock}:")
            print(f"  Portfolio: ${result['final_portfolio']:,.2f}")
            print(f"  Return: {result['return_pct']:+.2f}%")
            print(f"  Trades: {len(result['trades'])}")
        
        print(f"\n{'='*60}\n")
