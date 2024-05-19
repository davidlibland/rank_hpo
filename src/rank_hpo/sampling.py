"""Some sampling functions for the rank HPO project."""

import math
from typing import List

import numpy as np
import torch


def sorted_exp_sample(rates: List[float], order: List[float], n_samples=1):
    """
    Sample from a set of exponential distributions with different rates,
    given that they appear in a certain order.

    Args:
        rates: The rates of the exponential distributions.
        order: The order in which the samples appear.
        n_samples: The number of samples to draw.

    Returns:
        The samples.
    """
    order_inv = np.argsort(order)
    rates_sorted = rates[order_inv]
    rates_summed = np.cumsum(rates_sorted[::-1])[::-1]
    sample_shifts = np.random.exponential(
        1 / (rates_summed.reshape([1, -1]) + 1e-6), size=(n_samples, len(rates))
    )
    samples = np.cumsum(sample_shifts, axis=1)
    assert samples.shape == (n_samples, len(rates))
    return samples[:, order]


def sorted_exp_sample_torch(rates: torch.Tensor, order: torch.Tensor, n_samples=1):
    """
    Sample from a set of exponential distributions with different rates,
    given that they appear in a certain order.

    Args:
        rates: The rates of the exponential distributions.
        order: The order in which the samples appear.
        n_samples: The number of samples to draw.

    Returns:
        The samples.
    """
    order_inv = torch.argsort(order)
    rates_sorted = rates[order_inv]
    rates_summed = torch.cumsum(rates_sorted.flip(0), dim=0).flip(0)
    sample_shifts = (
        torch.distributions.exponential.Exponential(rates_summed)
        .sample((n_samples,))
        .to(rates)
    )
    samples = torch.cumsum(sample_shifts, dim=1)
    assert samples.shape == (n_samples, len(rates))
    return samples[:, order]


def langevin_step(
    theta: torch.Tensor, energy_func, step_size=1e-3, weight_decay=1e-4, temperature=1
):
    """
    Perform a Langevin step on a parameter theta, given an energy function.

    Args:
        theta: The parameter to update.
        energy_func: A function that takes theta as input and returns the energy.
        step_size: The step size.
        weight_decay: The weight decay.
        temperature: The temperature, lower temperatures correspond to less noise.
            A temperature of 1 corresponds to approximate sampling from the distribution.

    Returns:
        The updated parameter.
    """
    theta = theta.requires_grad_(True)
    if theta.grad is not None:
        theta.grad.zero_()
    energy = energy_func(theta)
    energy.backward()
    noise = torch.randn_like(theta)
    # if the grad is too large, clip it by shrinking the step size:
    if torch.norm(theta.grad) > 1:
        step_size /= torch.norm(theta.grad)
    theta.data -= (
        step_size * theta.grad + math.sqrt(2 * step_size) * noise * temperature
    )
    # Do weight decay:
    theta.data -= weight_decay * theta.data
    return theta, torch.linalg.norm(theta.grad)
