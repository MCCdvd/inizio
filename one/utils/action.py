import numpy as np
import pandas as pd


def to_discrete_action(action, threshold=0.33):
    if not isinstance(action, (list, tuple, pd.Series, np.ndarray)):
        return int(action)
    arr = np.array(action, dtype=np.float32).flatten()
    if arr.size == 0:
        return 0
    threshold = min(max(float(threshold), 0.0), 0.99)
    if arr.size == 1:
        scalar = float(arr[0])
        if scalar < -threshold:
            return 2
        if scalar > threshold:
            return 1
        return 0
    return int(arr.argmax())
