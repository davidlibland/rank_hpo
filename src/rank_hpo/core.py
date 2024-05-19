"""Main module."""

from typing import List, Callable, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import rankdata
from tqdm import tqdm

from rank_hpo.logging import TimeSeriesLogger
from rank_hpo.sampling import sorted_exp_sample, langevin_step, sorted_exp_sample_torch


def infer_rates(samples: List[np.ndarray], n_iter=1000):
    all_rates = []
    rates = np.ones(len(samples))
    flat_samples = np.concatenate(samples)
    order = rankdata(flat_samples, method="ordinal") - 1

    def flatten_rates(rates):
        return np.concatenate(
            [rates[i] * np.ones(len(samples[i])) for i in range(len(samples))]
        )

    for _ in range(n_iter):
        flat_rates = flatten_rates(rates)
        latent_samples = sorted_exp_sample(flat_rates, order, n_samples=1)[0, ...]
        new_rates = []
        start = 0
        end = len(samples[0])
        for i in range(len(samples)):
            new_rates.append(len(samples[i]) / (latent_samples[start:end].sum() + 1e-6))
            start = end
            end += len(samples[i])
        rates = np.array(new_rates)
        all_rates.append(rates)
    rate = np.stack(all_rates, axis=0).mean(axis=0)
    return rate / rate.max()


def _rank_data(data: torch.Tensor) -> torch.Tensor:
    """Returns the (zero-based) rank of elements in the data."""
    temp = torch.argsort(data.flatten())
    return torch.argsort(temp).reshape(data.shape)


def optimize_function_langevin(
    function: Callable[[torch.Tensor], torch.Tensor],
    x0s: List[torch.Tensor],
    log_rho,
    theta: torch.Tensor,
    n_evals=1000,
    n_iter=1000,
    weight_decay=1e-4,
    param_step_size=1e-3,
    sample_step_size=1e-3,
    n_langevin_steps=10,
    param_temperature_range=(1, 0.1),
    sample_temperature_range=(1, 0.1),
    n_log_samples=100,
    bounds=Tuple[torch.Tensor, torch.Tensor],
    batch_size=100
):
    x0s = [x0.clone().requires_grad_(False) for x0 in x0s]
    xs = torch.stack(x0s, dim=0)
    ys = torch.concat([function(x.detach()).flatten() for x in x0s])
    rates = torch.ones(len(xs), device=theta.device)
    theta_grad_logger = TimeSeriesLogger(n_log_samples)
    x_grad_logger = TimeSeriesLogger(n_log_samples)
    latent_samples_logger = TimeSeriesLogger(n_log_samples, dim=n_log_samples//10)
    for i in tqdm(range(n_evals)):
        temperature = (n_evals - i) / n_evals
        param_temperature = (
            param_temperature_range[0] - param_temperature_range[1]
        ) * temperature + param_temperature_range[1]
        sample_temperature = (
            sample_temperature_range[0] - sample_temperature_range[1]
        ) * temperature + sample_temperature_range[1]
        for _ in range(n_iter):
            order = _rank_data(ys)
            latent_samples = sorted_exp_sample_torch(rates+1e-6, order, n_samples=1).requires_grad_(False)
            latent_samples_logger.log(latent_samples.cpu())
            theta = theta.requires_grad_(True)
            for _ in range(n_langevin_steps):
                # Subsample the data:
                if len(xs) > batch_size:
                    ixs = torch.multinomial(torch.ones(len(xs)), batch_size, replacement=False)
                else:
                    ixs = torch.arange(len(xs))

                log_latent_samples = torch.log(latent_samples.flatten())
                log_latent_sample_scale = log_latent_samples.max()-log_latent_samples.min()
                log_latent_sample_scale = temperature* log_latent_sample_scale + (1-temperature)
                log_latent_samples = (log_latent_samples-log_latent_samples.min())/log_latent_sample_scale
                def energy_func(theta):
                    log_rho_xt = log_rho(xs[ixs].requires_grad_(False), theta).flatten()

                    return (
                        torch.exp(log_rho_xt+log_latent_samples.flatten()[ixs]) - 2*log_rho_xt
                        # The factor of 2 here comes from the change of variables
                        # from rho to log_rho.
                    ).sum()

                theta, grad = langevin_step(
                    theta,
                    energy_func,
                    step_size=param_step_size,
                    weight_decay=weight_decay,
                    temperature=param_temperature,
                )
                theta_grad_logger.log(grad.cpu())
            rates = torch.exp(log_rho(xs, theta)).detach().flatten()
        # Sample from the distribution:
        random_xs = torch.rand(300, xs.shape[1], device=xs.device, dtype=xs.dtype)
        random_xs = bounds[0] + random_xs * (bounds[1] - bounds[0])
        xs_options = torch.cat([xs, random_xs], dim=0)
        # Infer sampling rates for those ys:
        log_rates = log_rho(xs_options, theta).detach().flatten()
        # Sample according to the likelihood that we're near a good point:
        gumbels = -torch.log(-torch.log(1-torch.rand(len(log_rates)).to(log_rates)))
        ix = torch.argmax(log_rates + sample_temperature*gumbels)
        x = xs_options[ix].detach().view([1, -1]).clone()
        x = x.requires_grad_(True)

        def energy_function(x):
            return -log_rho(x, theta.requires_grad_(False))

        for _ in range(n_iter):
            x, grad = langevin_step(
                x,
                energy_function,
                step_size=sample_step_size,
                temperature=sample_temperature,
                weight_decay=0,
            )
            x_grad_logger.log(grad.cpu())
        y = function(x.detach())
        xs = torch.concatenate([xs, x.detach().flatten()[None, :]], dim=0)
        ys = torch.concatenate([ys, y.reshape(1)], dim=0)
        rates = torch.exp(log_rho(xs, theta)).detach().flatten()
    # Plot some diagnostics:
    latent_samples = sorted_exp_sample_torch(rates+1e-6, _rank_data(ys), n_samples=1).requires_grad_(False)
    ys_to_latents = torch.stack([ys.flatten(), latent_samples.flatten()], dim=1)
    _plot_diagnostics(latent_samples_logger, theta_grad_logger, x_grad_logger, ys_to_latents)
    return theta, xs, ys, latent_samples


def _plot_diagnostics(latent_samples_logger, theta_grad_logger, x_grad_logger, ys_to_latents):
    """Plot some helpful diagnostics."""
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    theta_grads = theta_grad_logger.get()
    axs[0, 0].plot(theta_grads[:, 0], theta_grads[:, 1], label="theta grads")
    axs[0, 0].set_title("theta grads")
    x_grads = x_grad_logger.get()
    axs[0, 1].plot(x_grads[:, 0], x_grads[:, 1], label="x grads")
    axs[0, 1].set_title("x grads")
    # Plot a scatterplot of latent samples:
    latent_samples = latent_samples_logger.get()
    pure_samples = latent_samples[:, 1:]
    sample_index = np.broadcast_to(latent_samples[:, :1], pure_samples.shape)
    axs[1, 0].scatter(sample_index.flatten(), pure_samples.flatten(), alpha=0.1)
    axs[1, 0].set_title("latent samples")
    axs[1, 0].set_yscale("log")
    axs[0, 0].set_xlabel("step")
    axs[0, 1].set_xlabel("step")
    axs[1, 0].set_xlabel("step")
    # Plot the relationship between y and latent samples:
    axs[1, 1].scatter(ys_to_latents[:, 0], ys_to_latents[:, 1], alpha=0.1)
    axs[1, 1].set_xlabel("true values")
    axs[1, 1].set_ylabel("latent values")
    axs[1, 1].set_yscale("log")
    axs[1, 1].set_title("true to latent values")
    fig.show()
