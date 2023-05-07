import numpy as np
import torch


def random_window(data, window_size):
    """Get window along 1st dimension of data.

    Args:
        data (Tensor): [time, features]
        window_size (int): how many time steps used to predict

    Returns:
        Tensor: [window_size, features]
        Tensor: [1]
    """
    window_size = window_size + 1
    start = np.random.randint(0, data.shape[0] - window_size)
    end = start + window_size
    return data[start:end - 1], data[end - 1]


def stock_selection(data):
    """Select random stock along 2nd dimension of data.

    Args:
        data (Tensor): [time, stock, features]

    Returns:
        Tensor: [time, features]
        Tensor: [1]
    """
    stock = np.random.randint(0, data.shape[1])
    return data[:, stock, :], torch.tensor(
        [stock], dtype=torch.long)  # [time, features], stock
