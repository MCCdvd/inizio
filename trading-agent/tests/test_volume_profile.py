import sys
import os
import numpy as np

# ensure local src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from trading_agent import VolumeProfileAnalyzer


def test_volume_profile_basic():
    # prices across 5 bins, volume peak at middle
    prices = np.array([10, 11, 12, 13, 14, 12, 12, 12, 13, 11])
    volumes = np.array([100, 100, 500, 100, 100, 300, 200, 400, 150, 50])

    analyzer = VolumeProfileAnalyzer(prices, volumes, bins=5)
    res = analyzer.calculate_profile()

    assert 'poc' in res and 'vah' in res and 'val' in res
    # POC should be near the price where volume concentrated (~12-13)
    assert 11.0 <= res['poc'] <= 13.5


def test_volume_profile_constant_prices():
    prices = np.array([10, 10, 10, 10])
    volumes = np.array([10, 20, 30, 40])
    analyzer = VolumeProfileAnalyzer(prices, volumes, bins=4)
    res = analyzer.calculate_profile()

    # In degenerate case, poc/vah/val should equal the single price
    assert res['poc'] == pytest.approx(10.0)
    assert res['vah'] == pytest.approx(10.0)
    assert res['val'] == pytest.approx(10.0)
