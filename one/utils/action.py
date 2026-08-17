import numpy as np
import pandas as pd


def to_discrete_action(action):
    if not isinstance(action, (list, tuple, pd.Series, np.ndarray)):
        return int(action)
    arr = np.array(action, dtype=np.float32).flatten()
    if arr.size == 0:
        return 0
    if arr.size == 1:
        scalar = float(arr[0])
        if scalar < -0.33:
            return 2
        if scalar > 0.33:
            return 1
        return 0
    return int(arr.argmax())
