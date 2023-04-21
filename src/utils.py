import pandas as pd
import numpy as np

def create_sliding_window_features(X, y, window_size=50):
    """
        X: DataFrame
        window_sizes: list of tuple [('col name', size), ..., ('col name', size)]
    """
    new_cols = [X.shift(i) for i in range(window_size+1)]
    new_X = pd.concat(new_cols, axis=1).dropna()
    start_index = np.intersect1d(y.index.tolist(), new_X.index.tolist()).min()

    return new_X.loc[y.loc[start_index].index.tolist()], y.loc[start_index:]
