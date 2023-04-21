import pandas as pd
import numpy as np

def create_sliding_window_features(X, y, window_sizes):
    """
        X: DataFrame
        window_sizes: list of tuple [('col name', size), ..., ('col name', size)]
    """
    for col, window_size in window_sizes:
        for i in range(window_size):
            X[f"{col}_lag_{i+1}"] = X[col].shift(i + 1)
    start_index = np.intersect1d(y.index.tolist(), X.dropna().index.tolist()).min()

    return X.dropna().loc[y.loc[start_index].index.tolist()], y.loc[start_index:]
