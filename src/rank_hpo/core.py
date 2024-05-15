"""Main module."""

from typing import List, Callable

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import rankdata

from rank_hpo.sampling import sorted_exp_sample, langevin_step


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


def optimize_function_langevin(
    function: Callable[[np.ndarray], float],
    x0s: List[np.ndarray],
    log_rho,
    theta: torch.Tensor,
    n_evals=1000,
    n_iter=1000,
    weight_decay=1e-4,
    param_step_size=1e-3,
    sample_step_size=1e-3,
    n_langevin_steps=10,
    batch_size=100,
    param_temperature_range=(1, 0.1),
    sample_temperature_range=(1, 0.1),
):
    x0s = [torch.tensor(x0, requires_grad=False) for x0 in x0s]
    xs = torch.stack(x0s, dim=0)
    ys = np.array([function(x.detach().numpy()).flatten() for x in x0s])
    rates = np.ones(len(xs))
    theta_grads = []
    x_grads = []
    latent_samples_log = []
    for i in range(n_evals):
        temperature = (n_evals - i) / n_evals
        param_temperature = (
            param_temperature_range[0] - param_temperature_range[1]
        ) * temperature + param_temperature_range[1]
        sample_temperature = (
            sample_temperature_range[0] - sample_temperature_range[1]
        ) * temperature + sample_temperature_range[1]
        for _ in range(n_iter):
            order = (
                rankdata(
                    np.nan_to_num(np.array(ys), nan=float("inf")), method="ordinal"
                )
                - 1
            )
            latent_samples = torch.tensor(
                sorted_exp_sample(rates, order, n_samples=1), requires_grad=False
            )
            latent_samples_log.append(latent_samples.flatten())
            theta = theta.requires_grad_(True)
            for _ in range(n_langevin_steps):
                # Choose a batch of xs:
                ixs = np.random.choice(
                    len(ys), size=min(batch_size, len(ys)), replace=False
                )

                def energy_func(theta):
                    log_rho_xt = log_rho(xs[ixs].requires_grad_(False), theta)
                    return (
                        torch.exp(log_rho_xt) * latent_samples[:, ixs] - log_rho_xt
                    ).sum()

                theta, grad = langevin_step(
                    theta,
                    energy_func,
                    step_size=param_step_size,
                    weight_decay=weight_decay,
                    temperature=param_temperature,
                )
                theta_grads.append(grad)
            rates = torch.exp(log_rho(xs, theta)).detach().numpy()
        # Sample from the distribution:
        # Seed x from the lowest temperature% of y values:
        n_ys = max(int(len(ys) * temperature), 1)
        best_ys = np.argsort(ys.flatten())[:n_ys]
        # Infer sampling rates for those ys:
        rates = log_rho(xs[best_ys], theta).detach().numpy()
        # Sample according to the likelihood that we're near a good point:
        ix = np.argmax(rates + np.random.gumbel(size=len(rates)))
        x = xs[ix].detach().view([1, -1]).clone()
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
            x_grads.append(grad)
        y = function(x.detach().numpy())
        xs = torch.concatenate([xs, x.detach().flatten()[None, :]], dim=0)
        ys = np.concatenate([ys, [y.flatten()]], axis=0)
        rates = torch.exp(log_rho(xs, theta)).detach().numpy()
    # Plot some diagnostics:
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].plot(theta_grads, label="theta grads")
    axs[0].set_title("theta grads")
    axs[1].plot(x_grads, label="x grads")
    axs[1].set_title("x grads")
    # Plot a scatterplot of latent samples:
    sample_index = torch.concatenate(
        [
            torch.ones_like(latent_samples_log[i]) * i
            for i in range(len(latent_samples_log))
        ],
        dim=0,
    )
    latent_samples_log = torch.concatenate(latent_samples_log, dim=0)
    axs[2].scatter(sample_index, torch.log(latent_samples_log), alpha=0.1)
    fig.show()
    return theta, xs, ys
