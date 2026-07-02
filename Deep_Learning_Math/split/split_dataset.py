import torch
import pandas as pd

def split_train_test(x, y, test_size=0.3):
    num_samples = len(x)
    indices = torch.randperm(num_samples)

    train_size = int(num_samples * (0.7))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    if type(x) == pd.DataFrame or type(x) == pd.Series:
        x_train = x.iloc[train_indices]
        y_train = y.iloc[train_indices]
        x_test = x.iloc[test_indices]
        y_test = y.iloc[test_indices]
    else:
        x_train = x[train_indices]
        y_train = y[train_indices]
        x_test = x[test_indices]
        y_test = y[test_indices]

    return x_train, y_train, x_test, y_test