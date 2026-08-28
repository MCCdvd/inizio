#!/bin/bash
# Run 7-run benchmark suite for all algorithms

cd "$(dirname "$0")"

echo "🚀 Starting 7-run benchmark suite..."
echo ""

# Default values
STOCK=${1:-AAPL}
EPISODES=${2:-250}
SHORT_SELLING=${3:-true}

# Show configuration
echo "Configuration:"
echo "  Stock: $STOCK"
echo "  Episodes: $EPISODES"
echo "  Short Selling: $SHORT_SELLING"
echo ""

# Run benchmark
if [ "$SHORT_SELLING" = "false" ]; then
    python benchmark_7runs.py --stock "$STOCK" --episodes "$EPISODES" --no-short-selling
else
    python benchmark_7runs.py --stock "$STOCK" --episodes "$EPISODES"
fi

echo ""
echo "✅ Benchmark complete!"
echo "   Results saved to: benchmarks/"
echo ""
