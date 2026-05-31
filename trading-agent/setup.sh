#!/bin/bash

# Setup script for trading-agent-ai

echo "Setting up trading-agent-ai..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo "Setup complete!"
echo "Activate environment with: source venv/bin/activate"
echo "Run training: python src/train.py --algorithm dqn --stock AAPL --episodes 50"
