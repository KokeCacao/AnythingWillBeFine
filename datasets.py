from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from settings import *


class StockDataset(torch.utils.data.Dataset):
    def __init__(self, series: Tensor, window_size: int):
        self.series = series  # [time, stock, features]
        self.window_size = window_size

    def __len__(self):
        return (self.series.shape[0] - self.window_size) * self.series.shape[1]

    def division_normalizer(self, X: Tensor, Y: Optional[Tensor]):
        first = X[0]
        X = X / first
        if Y is not None:
            Y = Y / first
        return X, Y

    def minmax_normalizer(self, X: Tensor, Y: Optional[Tensor], eps=1e-6):
        max = torch.max(X, dim=0, keepdim=False)[0]
        min = torch.min(X, dim=0, keepdim=False)[0]

        # Don't predit if max == min
        if torch.any(max == min):
            Y = min

        X = (X - min + eps) / (max - min + eps)
        if Y is not None:
            Y = (Y - min + eps) / (max - min + eps)
        return X, Y

    def __getitem__(self, idx):
        """Get window along 1st dimension of data.

        Args:
            idx (data index): 0 <= idx < len(self)

        Returns:
            Tensor: [window_size, features] - X
            Tensor: [features] - Y
            Tensor: [1] - stock_id
        """
        stock_id = idx % self.series.shape[1]
        idx = idx // self.series.shape[1]

        # generate data
        XY = self.series[idx:idx + self.window_size + 1, stock_id]
        X = XY[:-1]  # [window_size, features]
        Y = XY[-1]  # [features]
        stock_id = torch.tensor(stock_id, dtype=torch.long)

        # normalize
        X, Y = self.minmax_normalizer(X, Y)

        return X, Y, stock_id


def load_csv(ticker_list, format, start_date, end_date, suffix='_2.csv'):
    # load to pytorch tensor
    idx = pd.date_range(start_date, end_date)
    tensor = torch.empty((len(ticker_list), len(idx), len(format) - 1))
    for ticker in ticker_list:
        df = pd.read_csv(f'./data/{ticker}{suffix}')
        df = df[format]
        df['Date'] = pd.to_datetime(df['Date'])

        df = df.set_index('Date').reindex(idx, fill_value=np.NaN).reset_index()
        df = df[format[1:]]  # remove Date

        # linear interpolate fill NaN
        df = df.astype(float)
        df = df.interpolate(method='linear', limit_direction='both', axis=0)

        # forward fill and backward fill NaN
        # ffill: propagate last valid observation forward to next valid.
        # backfill / bfill: use next valid observation to fill gap.
        df = df.fillna(method='bfill')
        df = df.fillna(method='ffill')

        # average padding
        mean = df.mean(axis=0, skipna=True)  # for some data is NaN
        mean = mean.fillna(0)  # for all data is NaN
        df = df.fillna(mean)
        assert df.isnull().sum().sum(
        ) == 0, f"ticker: {ticker}, start_date: {start_date}, end_date: {end_date}"

        # To pytorch tensor
        tensor[ticker_list.index(ticker)] = torch.tensor(df.values)
    return tensor.transpose(0, 1)  # [time, stock, features]


def data_loading_fake():
    feaure = np.stack(
        [
            np.arange(0, 1000, 0.5),  # type: ignore
            np.arange(0, 1000, 0.5),  # type: ignore
            np.arange(0, 1000, 0.5),  # type: ignore
        ],
        axis=1)  # [time, features]
    # feaure = np.stack(
    #     [
    #         np.sin(np.arange(0, 1000, 0.5)),  # type: ignore
    #         np.cos(np.arange(0, 1000, 0.5)),  # type: ignore
    #         np.tan(np.arange(0, 1000, 0.5)),  # type: ignore
    #     ],
    # axis=1)  # [time, features]
    data = np.stack([feaure, feaure], axis=1)  # [time, stock, features]
    data = torch.from_numpy(data).float()
    return data


def train_test_split(series: torch.Tensor, n_test: int):
    return series[:-n_test], series[-n_test:]


def get_data():
    data = load_csv(TICKER_LIST, FORMAT, START_DATE,
                    END_DATE)  # [time, stock, features]

    train_series, test_series = train_test_split(data, n_test=N_TEST)
    n_features = data.shape[-1]

    return train_series, test_series, n_features