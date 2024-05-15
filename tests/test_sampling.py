import hypothesis
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from rank_hpo.sampling import sorted_exp_sample, sorted_exp_sample_torch, langevin_step

from hypothesis import given, strategies as st
from scipy.stats import rankdata


@given(
    st.lists(
        st.tuples(
            st.floats(
                min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False
            ),
            st.integers(min_value=0, max_value=100),
        ),
        min_size=2,
        unique_by=lambda x: x[1],
        max_size=10,
    ),
    st.integers(min_value=1, max_value=10),
)
def test_sorted_exp_sample(rates_orders, n_samples):
    """Test the sorted_exp_sample function."""
    rates, orders = zip(*rates_orders)
    orders = rankdata(orders, method="ordinal") - 1
    rates = np.array(rates)
    samples = sorted_exp_sample(rates, orders, n_samples)
    assert samples.shape == (n_samples, len(rates))
    assert (samples >= 0).all()
    for i in range(n_samples):
        assert (rankdata(samples[i], method="ordinal") - 1).tolist() == orders.tolist()


@given(
    st.lists(
        st.floats(min_value=1e-1, max_value=1e1, allow_nan=False, allow_infinity=False),
        min_size=2,
        unique=True,
        max_size=10,
    ),
)
@pytest.mark.parametrize("n_samples", [1000])
def test_sorted_exp_sample(n_samples, rates):
    """Test the sorted_exp_sample_torch function."""
    # orders = torch.arange(len(rates) - 1, -1, -1)
    orders = np.arange(len(rates))
    rates = np.array(rates)
    samples = sorted_exp_sample(rates, orders, n_samples)
    sample_diffs = np.diff(samples, axis=1, prepend=0)
    empirical_rates = 1 / np.mean(sample_diffs, axis=0)
    assert np.allclose(
        empirical_rates, np.cumsum(rates[::-1], axis=0)[::-1], atol=1, rtol=5e-1
    )


@given(
    st.lists(
        st.tuples(
            st.floats(
                min_value=1e-2, max_value=1e2, allow_nan=False, allow_infinity=False
            ),
            st.integers(min_value=0, max_value=100),
        ),
        min_size=2,
        unique_by=lambda x: x[1],
        max_size=10,
    ),
    st.integers(min_value=1, max_value=10),
)
def test_sorted_exp_sample_torch(rates_orders, n_samples):
    """Test the sorted_exp_sample_torch function."""
    rates, orders = zip(*rates_orders)
    orders = torch.tensor(rankdata(orders, method="ordinal") - 1)
    rates = torch.tensor(rates)
    samples = sorted_exp_sample_torch(rates, orders, n_samples)
    assert samples.shape == (n_samples, len(rates))
    assert (samples >= 0).all()
    for i in range(n_samples):
        assert (rankdata(samples[i], method="ordinal") - 1).tolist() == orders.tolist()


@given(
    st.lists(
        st.floats(min_value=0.1, max_value=10, allow_nan=False, allow_infinity=False),
        min_size=2,
        unique=True,
        max_size=10,
    ),
)
@pytest.mark.parametrize("n_samples", [1000])
def test_sorted_exp_sample_torch(n_samples, rates):
    """Test the sorted_exp_sample_torch function."""
    # orders = torch.arange(len(rates) - 1, -1, -1)
    orders = torch.arange(len(rates))
    rates = torch.tensor(rates)
    samples = sorted_exp_sample_torch(rates, orders, n_samples)
    sample_diffs = torch.diff(samples, dim=1, prepend=torch.zeros(n_samples, 1))
    empirical_rates = 1 / torch.mean(sample_diffs, dim=0)
    assert torch.allclose(
        empirical_rates, torch.cumsum(rates.flip(0), dim=0).flip(0), atol=1, rtol=5e-1
    )


@pytest.mark.skip
@hypothesis.settings(max_examples=10)
@given(
    st.floats(min_value=-10, max_value=5, allow_nan=False, allow_infinity=False).filter(
        lambda x: abs(x) > 1e-6
    ),
    st.floats(min_value=1e-1, max_value=3, allow_nan=False, allow_infinity=False),
)
@pytest.mark.parametrize("n_samples", [1000])
@pytest.mark.parametrize("n_steps", [100])
def test_langevin_step(n_samples, n_steps, mean, std):
    def energy_func(theta):
        return ((theta - mean) ** 2 / (2 * std**2)).sum()

    theta = torch.zeros(n_samples)
    for _ in range(n_steps):
        theta = langevin_step(theta, energy_func, 3e-2)
        assert torch.isfinite(theta).all()
        assert theta.shape == (n_samples,)
    assert (theta.mean() - mean) ** 2 / std**2 < 100 / n_samples
