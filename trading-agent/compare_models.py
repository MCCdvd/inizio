#!/usr/bin/env python3
"""
Model Comparison Framework for Trading Agent
Runs all 3 RL algorithms (DQN, PPO, A3C) with identical parameters and generates comparison report.
"""

import json
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from train import train_dqn_agent, train_ppo_agent, train_a3c_agent


class ModelComparator:
    """Framework for comparing RL trading agents."""

    def __init__(self, output_base: str, stock: str, episodes: int, flat_fee: float = 4.0, seed: int = 42):
        """
        Initialize comparator.
        
        Args:
            output_base: Base directory for all results
            stock: Stock symbol (e.g., 'AAPL')
            episodes: Number of training episodes
            flat_fee: Trading fee
            seed: Random seed for reproducibility
        """
        self.output_base = Path(output_base)
        self.stock = stock
        self.episodes = episodes
        self.flat_fee = flat_fee
        self.seed = seed
        self.results = {}
        self.output_base.mkdir(parents=True, exist_ok=True)

    def _get_model_output_dir(self, model_name: str) -> Path:
        """Get output directory for a specific model."""
        model_dir = self.output_base / f"{model_name}_{self.stock}_{self.episodes}ep"
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    def train_model(self, model_name: str, train_func) -> Dict:
        """
        Train a single model and collect results.
        
        Args:
            model_name: Name of the model (dqn, ppo, a3c)
            train_func: Training function to call
            
        Returns:
            Dictionary with training results
        """
        output_dir = self._get_model_output_dir(model_name)
        model_path = output_dir / "model.pt"

        print(f"\n{'='*70}")
        print(f"Training {model_name.upper()} Model")
        print(f"{'='*70}")
        print(f"Episodes: {self.episodes}")
        print(f"Stock: {self.stock}")
        print(f"Output Dir: {output_dir}")
        print(f"Flat Fee: ${self.flat_fee}")

        try:
            # Train the model
            train_func(
                stock_symbol=self.stock,
                episodes=self.episodes,
                seed=self.seed,
                output_dir=str(output_dir),
                save_model_path=str(model_path),
                flat_fee=self.flat_fee
            )

            # Load results
            summary_path = output_dir / "summary.json"
            if summary_path.exists():
                with open(summary_path, 'r') as f:
                    results = json.load(f)
                print(f"✅ {model_name.upper()} training completed successfully!")
                return results
            else:
                print(f"⚠️  Summary file not found for {model_name}")
                return None

        except Exception as e:
            print(f"❌ Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_all_models(self) -> Dict[str, Dict]:
        """Run all three models."""
        models = {
            'dqn': train_dqn_agent,
            'ppo': train_ppo_agent,
            'a3c': train_a3c_agent,
        }

        for model_name, train_func in models.items():
            self.results[model_name] = self.train_model(model_name, train_func)

        return self.results

    def generate_comparison_report(self) -> pd.DataFrame:
        """Generate comparison report from results."""
        if not self.results:
            print("No results to compare!")
            return None

        # Extract key metrics
        comparison_data = []
        for model_name, results in self.results.items():
            if results is None:
                continue

            row = {
                'Model': model_name.upper(),
                'Final Portfolio': f"${results.get('final_portfolio', 0):,.2f}",
                'Return %': f"{results.get('total_return_pct', 0):+.2f}%",
                'Sharpe Ratio': f"{results.get('sharpe_ratio', 0):.4f}",
                'Sortino Ratio': f"{results.get('sortino_ratio', 0):.4f}",
                'Max Drawdown': f"{results.get('max_drawdown', 0):.4f}",
                'Total Trades': results.get('total_trades', 0),
                'Win Rate %': f"{results.get('trade_metrics', {}).get('win_rate', 0):.2f}%",
                'Profit Factor': f"{results.get('trade_metrics', {}).get('profit_factor', 0):.2f}x",
                'Avg Win %': f"{results.get('trade_metrics', {}).get('avg_win', 0)*100:+.2f}%",
                'Avg Loss %': f"{results.get('trade_metrics', {}).get('avg_loss', 0)*100:+.2f}%",
                'Trades/Day': f"{results.get('activity_metrics', {}).get('trades_per_day', 0):.4f}",
            }
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)
        return df

    def save_comparison_report(self, df: pd.DataFrame, filename: str = "comparison_report.csv"):
        """Save comparison report to CSV."""
        report_path = self.output_base / filename
        df.to_csv(report_path, index=False)
        print(f"\n✅ Comparison report saved to: {report_path}")
        return report_path

    def print_comparison_summary(self):
        """Print formatted comparison summary."""
        df = self.generate_comparison_report()
        if df is None:
            return

        print("\n" + "="*70)
        print("MODEL COMPARISON SUMMARY")
        print("="*70)
        print(df.to_string(index=False))

        # Determine winner in key metrics
        print("\n" + "="*70)
        print("BEST PERFORMING MODELS")
        print("="*70)

        # Parse numeric values for comparison
        results_dict = {}
        for model_name, results in self.results.items():
            if results:
                results_dict[model_name] = {
                    'return': results.get('total_return_pct', 0),
                    'sharpe': results.get('sharpe_ratio', 0),
                    'sortino': results.get('sortino_ratio', 0),
                    'drawdown': results.get('max_drawdown', 0),  # Higher (less negative) is better
                }

        if results_dict:
            best_return = max(results_dict.items(), key=lambda x: x[1]['return'])
            best_sharpe = max(results_dict.items(), key=lambda x: x[1]['sharpe'])
            best_drawdown = max(results_dict.items(), key=lambda x: x[1]['drawdown'])

            print(f"🏆 Best Return: {best_return[0].upper()} ({best_return[1]['return']:+.2f}%)")
            print(f"🏆 Best Sharpe Ratio: {best_sharpe[0].upper()} ({best_sharpe[1]['sharpe']:.4f})")
            print(f"🏆 Best Drawdown Control: {best_drawdown[0].upper()} ({best_drawdown[1]['drawdown']:.4f})")

    def generate_html_report(self) -> str:
        """Generate HTML comparison report."""
        df = self.generate_comparison_report()
        if df is None:
            return None

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Trading Agent Model Comparison</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
        .report-info {{ margin: 20px 0; color: #666; }}
        table {{ border-collapse: collapse; width: 100%; background-color: white; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        th {{ background-color: #34495e; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 12px; border-bottom: 1px solid #ddd; }}
        tr:hover {{ background-color: #f9f9f9; }}
        .metric-group {{ margin: 20px 0; }}
        .footer {{ margin-top: 40px; color: #999; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Trading Agent Model Comparison Report</h1>
        <div class="report-info">
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Stock:</strong> {self.stock} | <strong>Episodes:</strong> {self.episodes} | <strong>Fee:</strong> ${self.flat_fee}</p>
        </div>
    </div>

    <div class="metric-group">
        <h2>Performance Metrics</h2>
        {df.to_html(index=False, border=0)}
    </div>

    <div class="footer">
        <p>Report generated by ModelComparator framework</p>
    </div>
</body>
</html>
"""
        html_path = self.output_base / "comparison_report.html"
        with open(html_path, 'w') as f:
            f.write(html_content)
        print(f"✅ HTML report saved to: {html_path}")
        return str(html_path)


def main():
    parser = argparse.ArgumentParser(description="Compare RL trading agent models")
    parser.add_argument('--stock', type=str, default='AAPL', help='Stock symbol')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--flat-fee', type=float, default=4.0, help='Trading fee')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='./model_comparison', help='Output directory')
    
    args = parser.parse_args()

    # Create comparator
    comparator = ModelComparator(
        output_base=args.output_dir,
        stock=args.stock,
        episodes=args.episodes,
        flat_fee=args.flat_fee,
        seed=args.seed
    )

    # Run all models
    print("\n" + "="*70)
    print("TRADING AGENT MODEL COMPARISON FRAMEWORK")
    print("="*70)
    print(f"Comparing 3 RL Algorithms: DQN, PPO, A3C")
    print(f"Parameters: Stock={args.stock}, Episodes={args.episodes}, Fee=${args.flat_fee}")
    print("="*70)

    comparator.run_all_models()

    # Generate reports
    comparator.print_comparison_summary()
    df = comparator.generate_comparison_report()
    if df is not None:
        comparator.save_comparison_report(df)
        comparator.generate_html_report()

    print("\n" + "="*70)
    print("Comparison complete! Check the output directory for detailed results.")
    print("="*70)


if __name__ == '__main__':
    main()
