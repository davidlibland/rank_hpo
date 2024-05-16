from rank_hpo.logging import TimeSeriesLogger

from hypothesis import given, strategies as st


@given(
    st.lists(st.integers(min_value=-10000, max_value=10000), min_size=1, max_size=1000),
    st.integers(min_value=1, max_value=1000),
)
def test_timeserieslogger(samples, max_samples):
    """Test the TimeSeriesLogger."""
    logger = TimeSeriesLogger(max_samples=max_samples)
    for x in samples:
        logger.log(x)
    logged = logger.get()
    n_logged = logged.shape[0]
    assert n_logged == min(len(samples), max_samples)
    assert len(set(logged[:, 0])) == n_logged, "Must have unique indices."
    for i in range(logged.shape[0]):
        ix = logged[i, 0]
        assert ix == int(ix)
        ix = int(ix)
        assert logged[i, 1] == samples[ix], "Logged value must match input."


@given(
    st.lists(
        st.tuples(
            st.integers(min_value=-10000, max_value=10000),
            st.integers(min_value=-10000, max_value=10000),
        ),
        min_size=1,
        max_size=1000,
    ),
    st.integers(min_value=1, max_value=1000),
)
def test_timeserieslogger_on_tuples(samples, max_samples):
    """Test the TimeSeriesLogger on tuples."""
    logger = TimeSeriesLogger(max_samples=max_samples, dim=2)
    for x in samples:
        logger.log(x)
    logged = logger.get()
    n_logged = logged.shape[0]
    assert n_logged == min(len(samples), max_samples)
    assert len(set(logged[:, 0])) == n_logged, "Must have unique indices."
    for i in range(logged.shape[0]):
        ix = logged[i, 0]
        assert ix == int(ix)
        ix = int(ix)
        assert logged[i, 1:].tolist() == list(
            samples[ix]
        ), "Logged value must match input."
