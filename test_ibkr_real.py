#!/usr/bin/env python
"""
Real-world test script for IBKR Data Connector.

This script demonstrates how to use the new IBKR integration with the TradingEnvironmentWithVolumeProfile.

Prerequisites:
    1. IB Gateway or TWS must be running on 127.0.0.1:7497 with API enabled
    2. ib-insync >= 0.9.86 must be installed: pip install ib-insync
    3. Python 3.8+

Usage:
    python test_ibkr_real.py
"""

import sys
import os
import logging
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "trading-agent", "src"))

# Configure logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import after path is set
from trading_agent import TradingEnvironmentWithVolumeProfile
import numpy as np

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")

def test_yahoo_finance():
    """Test 1: Yahoo Finance (default) - should always work."""
    print_section("TEST 1: Yahoo Finance (Baseline)")
    
    try:
        # Create environment with default Yahoo Finance
        env = TradingEnvironmentWithVolumeProfile(
            stock_symbol="AAPL",
            initial_balance=10000,
            data_source="yahoo"
        )
        
        # Load 6 months of historical data
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
        
        logger.info(f"Loading AAPL from Yahoo Finance: {start_date} to {end_date}")
        prices, volumes = env.load_data(start_date, end_date)
        
        if len(prices) == 0:
            print("❌ FAILED: No data returned")
            return False
        
        print(f"✅ SUCCESS: Yahoo Finance")
        print(f"   • Bars fetched: {len(prices)}")
        print(f"   • Price range: ${prices.min():.2f} - ${prices.max():.2f}")
        print(f"   • Current price: ${prices[-1]:.2f}")
        print(f"   • Avg volume: {volumes.mean():.0f}")
        print(f"   • Data source: {env.data_source}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        logger.exception("Yahoo Finance test failed")
        return False


def test_ibkr_direct():
    """Test 2: Interactive Brokers (direct connection)."""
    print_section("TEST 2: Interactive Brokers (Direct Connection)")
    
    print("📋 Prerequisites:")
    print("   1. IB Gateway running on 127.0.0.1:7497")
    print("   2. Socket client enabled in IB Gateway settings")
    print("   3. ib-insync installed: pip install ib-insync")
    
    try:
        # Check if ib-insync is installed
        import ib_insync
        print("✅ ib-insync is installed")
    except ImportError:
        print("❌ ib-insync not installed!")
        print("   Install with: pip install ib-insync>=0.9.86")
        return False
    
    try:
        # Create environment with IBKR
        env = TradingEnvironmentWithVolumeProfile(
            stock_symbol="AAPL",
            initial_balance=10000,
            data_source="ibkr",
            ibkr_host="127.0.0.1",
            ibkr_port=7497,
            ibkr_client_id=11,
            ibkr_timeframe="1 day"
        )
        
        # Load 6 months of historical data
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
        
        logger.info(f"Loading AAPL from IBKR: {start_date} to {end_date}")
        prices, volumes = env.load_data(start_date, end_date)
        
        if len(prices) == 0:
            print("❌ FAILED: No data returned")
            return False
        
        print(f"✅ SUCCESS: Interactive Brokers")
        print(f"   • Bars fetched: {len(prices)}")
        print(f"   • Price range: ${prices.min():.2f} - ${prices.max():.2f}")
        print(f"   • Current price: ${prices[-1]:.2f}")
        print(f"   • Avg volume: {volumes.mean():.0f}")
        print(f"   • Data source: {env.data_source}")
        print(f"   • IB Gateway: {env.ibkr_host}:{env.ibkr_port}")
        
        return True
        
    except ConnectionRefusedError as e:
        print(f"❌ FAILED: Connection refused to IB Gateway")
        print(f"   Error: {e}")
        print(f"\n   ⚠️  IB Gateway is not running. Next test will verify fallback...")
        return False
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        logger.exception("IBKR test failed")
        return False


def test_ibkr_with_fallback():
    """Test 3: IBKR with automatic fallback to Yahoo Finance."""
    print_section("TEST 3: IBKR with Fallback to Yahoo Finance")
    
    print("Scenario: IB Gateway is offline/unavailable")
    print("Expected: System should gracefully fallback to Yahoo Finance\n")
    
    try:
        # Create environment with IBKR but with wrong port (to simulate connection failure)
        env = TradingEnvironmentWithVolumeProfile(
            stock_symbol="MSFT",
            initial_balance=10000,
            data_source="ibkr",
            ibkr_host="127.0.0.1",
            ibkr_port=9999,  # Wrong port - will fail and fallback
            ibkr_client_id=11
        )
        
        # Load data
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
        
        logger.info(f"Loading MSFT with IBKR fallback: {start_date} to {end_date}")
        prices, volumes = env.load_data(start_date, end_date)
        
        if len(prices) == 0:
            print("❌ FAILED: No data returned even after fallback")
            return False
        
        print(f"✅ SUCCESS: Fallback worked correctly")
        print(f"   • Bars fetched: {len(prices)}")
        print(f"   • Price range: ${prices.min():.2f} - ${prices.max():.2f}")
        print(f"   • Current price: ${prices[-1]:.2f}")
        print(f"   • ⚠️  Fell back to Yahoo Finance (original source: IBKR)")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        logger.exception("Fallback test failed")
        return False


def test_backward_compatibility():
    """Test 4: Backward compatibility - old API still works."""
    print_section("TEST 4: Backward Compatibility")
    
    print("Testing: TradingEnvironmentWithVolumeProfile without data_source parameter\n")
    
    try:
        # Old API - no data_source parameter
        env = TradingEnvironmentWithVolumeProfile("GOOGL")
        
        # Should default to "yahoo"
        if env.data_source != "yahoo":
            print(f"❌ FAILED: Expected default data_source='yahoo', got '{env.data_source}'")
            return False
        
        # Load data to verify it works
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        prices, volumes = env.load_data(start_date, end_date)
        
        if len(prices) == 0:
            print("❌ FAILED: No data returned")
            return False
        
        print(f"✅ SUCCESS: Backward compatibility maintained")
        print(f"   • data_source defaults to: '{env.data_source}'")
        print(f"   • Bars fetched: {len(prices)}")
        print(f"   • Old API works seamlessly ✓")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        logger.exception("Backward compatibility test failed")
        return False


def test_multiple_symbols():
    """Test 5: Fetch data for multiple symbols."""
    print_section("TEST 5: Multiple Symbols")
    
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
    
    results = {}
    
    for symbol in symbols:
        try:
            env = TradingEnvironmentWithVolumeProfile(symbol)
            prices, volumes = env.load_data(start_date, end_date)
            
            if len(prices) > 0:
                results[symbol] = {
                    "bars": len(prices),
                    "price": prices[-1],
                    "min": prices.min(),
                    "max": prices.max()
                }
                print(f"✅ {symbol}: {len(prices)} bars, ${prices[-1]:.2f}")
            else:
                print(f"⚠️  {symbol}: No data")
                
        except Exception as e:
            print(f"❌ {symbol}: {e}")
    
    if len(results) > 0:
        print(f"\n✅ SUCCESS: Fetched data for {len(results)}/{len(symbols)} symbols")
        return True
    else:
        print(f"\n❌ FAILED: Could not fetch any data")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("  IBKR Data Connector - Real-World Test Suite")
    print("  PR #8: feat: IBKR data connector with modular connector architecture")
    print("=" * 70)
    
    results = {}
    
    # Test 1: Yahoo Finance (baseline)
    results["Yahoo Finance"] = test_yahoo_finance()
    
    # Test 2: IBKR (direct connection)
    results["IBKR Direct"] = test_ibkr_direct()
    
    # Test 3: IBKR with fallback
    results["IBKR Fallback"] = test_ibkr_with_fallback()
    
    # Test 4: Backward compatibility
    results["Backward Compat"] = test_backward_compatibility()
    
    # Test 5: Multiple symbols
    results["Multiple Symbols"] = test_multiple_symbols()
    
    # Summary
    print_section("TEST SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! The IBKR integration is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review the logs above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
