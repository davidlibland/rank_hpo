"""Some standard distributions for hyperparameter inference."""

from typing import Literal, Union

import numpy as np
import torch

LEARN = Literal["learn"]


def gaussian(std: Union[float, LEARN] = 1):
    """Returns a simple gaussian distribution, with mobile mean"""

    def log_rho_fixed_std(x, theta):
        return -0.5 * torch.sum((x - theta.flatten()[None, :]) ** 2, dim=1) / std**2

    def log_rho_learn_std(x, theta):
        # Theta is a 2n-dimensional vector, the first n dimensions
        # are the mean of the distribution, the second n-dimensions are the
        # diagonal standard deviations of the distribution.
        n = len(theta) // 2
        mean, log_std = theta[:n].reshape(1, -1), theta[n:].reshape(1, -1)
        var = torch.sigmoid(2 * log_std)
        return -torch.sum(0.5 * (x - mean) ** 2 / var, dim=1) - torch.sum(
            log_std, dim=1
        )

    if std == "learn":
        return log_rho_learn_std
    else:
        return log_rho_fixed_std


def two_layer_mlp(visible_dim):
    def log_rho(x, theta):
        # Theta is a 2-visible_dim-dimensional vector, the first visible_dim dimensions
        # are the mean of the distribution, the second visible_dim-dimensions are the
        # diagonal standard deviations of the distribution.
        mean, log_std = theta[:visible_dim].reshape(1, -1), theta[visible_dim:].reshape(
            1, -1
        )
        var = torch.sigmoid(2 * log_std)
        return -torch.sum(0.5 * (x - mean) ** 2 / var, dim=1) - torch.sum(
            log_std, dim=1
        )

    return log_rho


def gaussian_mixture(grid: np.ndarray, std=1):
    """
    Builds a gaussian mixture model with gaussians centered at the
    grid points.

    Args:
        grid: The grid of points to center the gaussians at, of shape (n, d)
    """
    grid = torch.tensor(grid, dtype=torch.double, requires_grad=False)

    def log_rho(x, theta):
        """Theta is of shape (n, d), and represents the weight of each gaussian"""
        gaussians = (
            -0.5 * ((x[:, None, :] - grid[None, :, :]) ** 2).sum(dim=2) / (std**2)
            + theta[None, :]
        )
        result = torch.logsumexp(gaussians, dim=1)
        # print(x.shape, grid.shape, theta.shape, result.shape)
        return result

    return log_rho
