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


def n_layer_mlp(visible_dim, reg=1e-4, n_layers=2):
    """
    Builds a simple n-layer mlp with gelu activations and layer normalization.

    Args:
        visible_dim: The dimension of the input
        reg: The regularization strength
        n_layers: The number of layers in the mlp

    Returns:
        A function that computes the log density of the distribution
    """

    def log_rho(x, theta):
        """
        This is a simple 2-layer mlp units and gelu activation
        x must be of shape [n_samples, visible_dim]
        theta must be of shape [2 * (visible_dim + 1) + (n_layers-2) * (hidden_dim + 1), hidden_dim]
        """
        assert (
            x.ndim == 2
        ), f"x should be a 2d tensor, or shape (n_samples, d), got shape {x.shape}"
        assert theta.shape[0] == 2 * (visible_dim + 1) + (n_layers - 2) * (
            theta.shape[1] + 1
        ), (
            f"theta should have the right number of elements, "
            f"got {theta.shape[0]} instead of "
            f"{2 * (visible_dim + 1) + (n_layers - 2) * (theta.shape[1] + 1)}"
        )
        # Peel the weights and biases from theta
        layer_1_weights = theta[:visible_dim]
        mean = theta[visible_dim : 2 * visible_dim][:, 0].reshape(1, -1)
        layer_1_bias = theta[2 * visible_dim : 2 * visible_dim + 1]
        final_weights = theta[2 * visible_dim + 1 : 2 * visible_dim + 2]

        # Apply the first layer, this embeds the input in a hidden_dim-dimensional space
        y = x @ layer_1_weights + layer_1_bias
        y = torch.nn.functional.layer_norm(
            y, normalized_shape=layer_1_weights.shape[1:]
        )
        y = torch.nn.functional.gelu(y)
        dim = theta.shape[1]
        for i in range(0, n_layers - 2):
            # Iteratively peel off the weights and biases
            start = 2 * (visible_dim + 1) + i * (dim + 1)
            weights = theta[start : start + dim]
            bias = theta[start + dim : start + dim + 1]
            # Apply the layer
            y_ = y @ weights + bias
            y_ = torch.nn.functional.layer_norm(y_, normalized_shape=y.shape[1:])
            y_ = torch.nn.functional.gelu(y_)
            y = y_ + y
        # Apply the final layer, projecting to 1-dimension
        out = y @ final_weights.T
        prior = (reg * (x - mean) ** 2).sum(dim=1, keepdim=True)
        return out - prior

    return log_rho


def gaussian_mixture(grid: np.ndarray, std=1):
    """
    Builds a gaussian mixture model with gaussians centered at the
    grid points.

    Args:
        grid: The grid of points to center the gaussians at, of shape (n, d)
        std: The standard deviation of each component.
    """

    def log_rho(x, theta):
        """Theta is of shape (n, d), and represents the weight of each gaussian"""
        assert (
            x.ndim == 2
        ), f"x should be a 2d tensor, or shape (n_samples, d), got shape {x.shape}"
        assert (
            theta.ndim == 1
        ), f"theta should be a 1d tensor, or shape (n_gaussians), got shape {theta.shape}"
        assert (
            grid.shape[1] == x.shape[1]
        ), f"grid should have the same number of columns as x, got {grid.shape[1]} and {x.shape[1]}"
        assert (
            theta.shape[0] == grid.shape[0]
        ), f"theta should have the same number of elements as the number of gaussians, got {theta.shape[0]} and {grid.shape[0]}"
        gaussians = (
            -0.5 * ((x[:, None, :] - grid[None, :, :]) ** 2).sum(dim=2) / (std**2)
            + theta[None, :]
        )
        result = torch.logsumexp(gaussians, dim=1)
        # print(x.shape, grid.shape, theta.shape, result.shape)
        return result

    return log_rho
