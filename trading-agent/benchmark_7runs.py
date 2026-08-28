#!/usr/bin/env python3
"""
Robust benchmarking framework with 7 runs per algorithm.
Runs each algorithm multiple times with different seeds to establish
statistically significant performance baselines.
"""

import os
import sys
import csv
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any
import statistics

# Add src to path
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))


@dataclass
class BenchmarkResult:
    """Single algorithm run result."""
    algorithm: str
    seed: int
    return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    total_trades: int
    win_rate_pct: float
    profit_factor: float
    avg_win_pct: float
    avg_loss_pct: float
    trades_per_day: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'Algorithm': self.algorithm,
            'Seed': self.seed,
            'Return %': f"{self.return_pct:+.2f}%",
            'Sharpe Ratio': f"{self.sharpe_ratio:.4f}",
            'Sortino Ratio': f"{self.sortino_ratio:.4f}",
            'Max Drawdown': f"{self.max_drawdown:.4f}",
            'Total Trades': self.total_trades,
            'Win Rate %': f"{self.win_rate_pct:.2f}%",
            'Profit Factor': f"{self.profit_factor:.2f}x",
            'Avg Win %': f"{self.avg_win_pct:+.2f}%",
            'Avg Loss %': f"{self.avg_loss_pct:+.2f}%",
            'Trades/Day': f"{self.trades_per_day:.4f}",
        }


@dataclass
class AggregatedStats:
    """Aggregated statistics for one algorithm across multiple runs."""
    algorithm: str
    num_runs: int
    
    # Return metrics
    return_mean: float
    return_std: float
    return_min: float
    return_max: float
    return_ci_lower: float
    return_ci_upper: float
    
    # Sharpe metrics
    sharpe_mean: float
    sharpe_std: float
    
    # Other key metrics
    trades_mean: float
    win_rate_mean: float
    profit_factor_mean: float
    
    def to_dict(self) -> Dict[str, str]:
        return {
            'Algorithm': self.algorithm,
            'Runs': str(self.num_runs),
            'Return Mean': f"{self.return_mean:+.2f}%",
            'Return Std': f"{self.return_std:.2f}%",
            'Return Min': f"{self.return_min:+.2f}%",
            'Return Max': f"{self.return_max:+.2f}%",
            'Return CI (95%)': f"[{self.return_ci_lower:+.2f}%, {self.return_ci_upper:+.2f}%]",
            'Sharpe Mean': f"{self.sharpe_mean:.4f}",
            'Sharpe Std': f"{self.sharpe_std:.4f}",
            'Trades Mean': f"{self.trades_mean:.1f}",
            'Win Rate Mean': f"{self.win_rate_mean:.2f}%",
            'Profit Factor Mean': f"{self.profit_factor_mean:.2f}x",
        }


class BenchmarkRunner:
    """Orchestrates multi-run benchmarking."""
    
    def __init__(self, stock: str = "AAPL", episodes: int = 250, 
                 seeds: List[int] = None, short_selling: bool = True):
        self.stock = stock
        self.episodes = episodes
        self.seeds = seeds or [42, 123, 456, 789, 1337, 2024, 99999]
        self.short_selling = short_selling
        self.algorithms = ["DQN", "PPO", "A3C"]
        self.results: List[BenchmarkResult] = []
        
    def run_training(self, algorithm: str, seed: int) -> BenchmarkResult:
        """Run a single training session and extract results."""
        print(f"  [{algorithm} | Seed {seed}] Starting training...")
        
        # Build command
        cmd = [
            "python", "src/train.py",
            "--algorithm", algorithm,
            "--stock", self.stock,
            "--episodes", str(self.episodes),
            "--seed", str(seed),
        ]
        
        if self.short_selling:
            cmd.append("--short-selling")
        
        try:
            # Run training
            result = subprocess.run(
                cmd,
                cwd=str(SCRIPT_DIR),
                capture_output=True,
                text=True,
                timeout=1200  # 20 minute timeout
            )
            
            if result.returncode != 0:
                print(f"    ⚠️  Training failed: {result.stderr[:200]}")
                # Return zeros for failed run
                return BenchmarkResult(
                    algorithm=algorithm, seed=seed,
                    return_pct=0, sharpe_ratio=0, sortino_ratio=0,
                    max_drawdown=0, total_trades=0, win_rate_pct=0,
                    profit_factor=0, avg_win_pct=0, avg_loss_pct=0,
                    trades_per_day=0
                )
            
            # Parse output to extract metrics
            output_lines = result.stdout.strip().split('\n')
            metrics = self._parse_output(output_lines, algorithm)
            
            print(f"    ✓ Return: {metrics['return_pct']:+.2f}%, Sharpe: {metrics['sharpe_ratio']:.4f}")
            
            return BenchmarkResult(
                algorithm=algorithm,
                seed=seed,
                **metrics
            )
            
        except subprocess.TimeoutExpired:
            print(f"    ⚠️  Training timeout")
            return BenchmarkResult(
                algorithm=algorithm, seed=seed,
                return_pct=0, sharpe_ratio=0, sortino_ratio=0,
                max_drawdown=0, total_trades=0, win_rate_pct=0,
                profit_factor=0, avg_win_pct=0, avg_loss_pct=0,
                trades_per_day=0
            )
    
    def _parse_output(self, lines: List[str], algorithm: str) -> Dict[str, Any]:
        """Extract metrics from training output."""
        metrics = {
            'return_pct': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'total_trades': 0,
            'win_rate_pct': 0.0,
            'profit_factor': 0.0,
            'avg_win_pct': 0.0,
            'avg_loss_pct': 0.0,
            'trades_per_day': 0.0,
        }
        
        # Look for JSON output or formatted metrics
        for line in lines:
            line_lower = line.lower()
            
            # Try to parse JSON if present
            if line.strip().startswith('{'):
                try:
                    data = json.loads(line)
                    if 'final_return_pct' in data:
                        metrics['return_pct'] = float(data.get('final_return_pct', 0))
                    if 'sharpe_ratio' in data:
                        metrics['sharpe_ratio'] = float(data.get('sharpe_ratio', 0))
                    if 'sortino_ratio' in data:
                        metrics['sortino_ratio'] = float(data.get('sortino_ratio', 0))
                    if 'max_drawdown' in data:
                        metrics['max_drawdown'] = float(data.get('max_drawdown', 0))
                    if 'total_trades' in data:
                        metrics['total_trades'] = int(data.get('total_trades', 0))
                    if 'win_rate_pct' in data:
                        metrics['win_rate_pct'] = float(data.get('win_rate_pct', 0))
                    if 'profit_factor' in data:
                        metrics['profit_factor'] = float(data.get('profit_factor', 0))
                    if 'avg_win_pct' in data:
                        metrics['avg_win_pct'] = float(data.get('avg_win_pct', 0))
                    if 'avg_loss_pct' in data:
                        metrics['avg_loss_pct'] = float(data.get('avg_loss_pct', 0))
                    if 'trades_per_day' in data:
                        metrics['trades_per_day'] = float(data.get('trades_per_day', 0))
                except:
                    pass
            
            # Parse text format lines
            if 'return' in line_lower and '%' in line:
                try:
                    val = float(line.split(':')[-1].strip().rstrip('%'))
                    metrics['return_pct'] = val
                except:
                    pass
            elif 'sharpe' in line_lower:
                try:
                    val = float(line.split(':')[-1].strip())
                    metrics['sharpe_ratio'] = val
                except:
                    pass
            elif 'total trades' in line_lower:
                try:
                    val = int(line.split(':')[-1].strip())
                    metrics['total_trades'] = val
                except:
                    pass
        
        return metrics
    
    def run_all_benchmarks(self) -> None:
        """Run benchmarks for all algorithms and seeds."""
        print(f"\n🚀 Starting 7-run benchmark")
        print(f"   Stock: {self.stock}, Episodes: {self.episodes}")
        print(f"   Algorithms: {', '.join(self.algorithms)}")
        print(f"   Seeds: {self.seeds}")
        print(f"   Short Selling: {'Enabled' if self.short_selling else 'Disabled'}\n")
        
        total_runs = len(self.algorithms) * len(self.seeds)
        current_run = 0
        
        for algorithm in self.algorithms:
            print(f"📊 {algorithm}:")
            for seed in self.seeds:
                current_run += 1
                print(f"  Run {current_run}/{total_runs}")
                result = self.run_training(algorithm, seed)
                self.results.append(result)
        
        print(f"\n✅ All benchmarks complete ({total_runs} runs)")
    
    def save_results(self, output_dir: str = None) -> str:
        """Save all results to CSV file."""
        if output_dir is None:
            output_dir = str(SCRIPT_DIR / "benchmarks")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = output_path / f"benchmark_results_{timestamp}.csv"
        
        # Write individual run results
        with open(csv_file, 'w', newline='') as f:
            fieldnames = [
                'Algorithm', 'Seed', 'Return %', 'Sharpe Ratio', 'Sortino Ratio',
                'Max Drawdown', 'Total Trades', 'Win Rate %', 'Profit Factor',
                'Avg Win %', 'Avg Loss %', 'Trades/Day'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                writer.writerow(result.to_dict())
        
        print(f"\n💾 Individual results saved: {csv_file}")
        return str(csv_file)
    
    def compute_statistics(self) -> Dict[str, AggregatedStats]:
        """Compute aggregated statistics per algorithm."""
        stats_by_algo = {}
        
        for algorithm in self.algorithms:
            algo_results = [r for r in self.results if r.algorithm == algorithm]
            
            if not algo_results:
                continue
            
            # Extract values
            returns = [r.return_pct for r in algo_results]
            sharpes = [r.sharpe_ratio for r in algo_results]
            trades = [r.total_trades for r in algo_results]
            win_rates = [r.win_rate_pct for r in algo_results]
            profit_factors = [r.profit_factor for r in algo_results]
            
            # Compute statistics
            n = len(returns)
            return_mean = statistics.mean(returns)
            return_std = statistics.stdev(returns) if n > 1 else 0
            
            # 95% confidence interval (t-distribution, df=n-1)
            # Simplified: use 1.96 * std / sqrt(n) for large samples
            ci_margin = 1.96 * return_std / (n ** 0.5) if return_std > 0 else 0
            
            sharpe_mean = statistics.mean(sharpes)
            sharpe_std = statistics.stdev(sharpes) if n > 1 else 0
            
            stats = AggregatedStats(
                algorithm=algorithm,
                num_runs=n,
                return_mean=return_mean,
                return_std=return_std,
                return_min=min(returns),
                return_max=max(returns),
                return_ci_lower=return_mean - ci_margin,
                return_ci_upper=return_mean + ci_margin,
                sharpe_mean=sharpe_mean,
                sharpe_std=sharpe_std,
                trades_mean=statistics.mean(trades),
                win_rate_mean=statistics.mean(win_rates),
                profit_factor_mean=statistics.mean(profit_factors),
            )
            
            stats_by_algo[algorithm] = stats
        
        return stats_by_algo
    
    def save_statistics(self, stats: Dict[str, AggregatedStats], 
                       output_dir: str = None) -> str:
        """Save aggregated statistics to CSV."""
        if output_dir is None:
            output_dir = str(SCRIPT_DIR / "benchmarks")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = output_path / f"benchmark_statistics_{timestamp}.csv"
        
        with open(csv_file, 'w', newline='') as f:
            fieldnames = list(list(stats.values())[0].to_dict().keys()) if stats else []
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for algo_stats in stats.values():
                writer.writerow(algo_stats.to_dict())
        
        print(f"📊 Statistics saved: {csv_file}")
        return str(csv_file)
    
    def print_summary(self, stats: Dict[str, AggregatedStats]) -> None:
        """Print summary statistics to console."""
        print("\n" + "="*90)
        print("📈 BENCHMARK SUMMARY (7 Runs Per Algorithm)")
        print("="*90)
        
        for algorithm in self.algorithms:
            if algorithm not in stats:
                continue
            
            s = stats[algorithm]
            print(f"\n{algorithm}:")
            print(f"  Return:        {s.return_mean:+7.2f}% ± {s.return_std:.2f}% "
                  f"(range: {s.return_min:+.2f}% to {s.return_max:+.2f}%)")
            print(f"  Return CI:     [{s.return_ci_lower:+.2f}%, {s.return_ci_upper:+.2f}%] (95%)")
            print(f"  Sharpe Ratio:  {s.sharpe_mean:7.4f} ± {s.sharpe_std:.4f}")
            print(f"  Trades:        {s.trades_mean:7.1f} avg")
            print(f"  Win Rate:      {s.win_rate_mean:7.2f}%")
            print(f"  Profit Factor: {s.profit_factor_mean:7.2f}x")
        
        print("\n" + "="*90)
        print("💡 Interpretation:")
        print("   - Lower std dev = more consistent algorithm")
        print("   - Higher Sharpe = better risk-adjusted returns")
        print("   - CI = where true mean likely lies (95% confidence)")
        print("="*90 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark trading algorithms with 7 runs per algorithm"
    )
    parser.add_argument("--stock", default="AAPL", help="Stock symbol (default: AAPL)")
    parser.add_argument("--episodes", type=int, default=250, 
                       help="Training episodes (default: 250)")
    parser.add_argument("--no-short-selling", action="store_true",
                       help="Disable short selling (default: enabled)")
    parser.add_argument("--seeds", type=int, nargs="+",
                       help="Custom seeds (default: 42 123 456 789 1337 2024 99999)")
    
    args = parser.parse_args()
    
    seeds = args.seeds or [42, 123, 456, 789, 1337, 2024, 99999]
    
    runner = BenchmarkRunner(
        stock=args.stock,
        episodes=args.episodes,
        seeds=seeds,
        short_selling=not args.no_short_selling
    )
    
    # Run all benchmarks
    runner.run_all_benchmarks()
    
    # Save results
    runner.save_results()
    
    # Compute and save statistics
    stats = runner.compute_statistics()
    runner.save_statistics(stats)
    
    # Print summary
    runner.print_summary(stats)


if __name__ == "__main__":
    main()
