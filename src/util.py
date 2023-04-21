import pandas as pd

def create_sliding_window_features(data, window_size, target_col):
    """
        data: DataFrame
        target_col: col name
    """
    for i in range(window_size):
        data[f"lag_{i+1}"] = data[target_col].shift(i + 1)
    data = data.dropna()
    return data
