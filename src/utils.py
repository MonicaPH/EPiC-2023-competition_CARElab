import pandas as pd
import numpy as np

def create_sliding_window_features(X, y, window_size, target_col):
    """
        X: DataFrame
        target_col: col name
    """
    for i in range(window_size):
        X[f"lag_{i+1}"] = X[target_col].shift(i + 1)
    start_index = np.intersect1d(y.index.tolist(), X.dropna().index.tolist()).min()

    return X.dropna().loc[y.loc[start_index].index.tolist()], y.loc[start_index:]
