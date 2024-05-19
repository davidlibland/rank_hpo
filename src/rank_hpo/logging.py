import numpy as np


class TimeSeriesLogger:
    """Logs a time series of vectors, subsampling over time if necessary."""

    def __init__(self, max_samples, dim=1):
        """Initializes the logger.

        Args:
            max_samples: The maximum number of samples (in time) to store.
            dim: The dimension of the vectors to log.
        """
        self.max_samples = max_samples
        self.total_samples = 0
        self.dim = dim
        self.storage = np.full((max_samples, 1 + dim), np.nan)
        # The first column stores the time, the second column stores the value.

    def log(self, x: np.ndarray):
        """
        Logs a new value, subsampling if necessary. Subsampling is done
        uniformly at random in time to ensure that no more than max_samples
        are stored.
        """
        self.total_samples += 1
        x = np.array(x).reshape(-1)
        if x.shape[0] != self.dim:
            # Resample uniformly:
            x = np.random.choice(x, self.dim, replace=True)
        row = np.concatenate(([self.total_samples - 1], x))
        if self.total_samples > self.max_samples:
            # Uniformly subsample the data:
            append_prob = self.max_samples / self.total_samples
            if np.random.rand() < append_prob:
                # Choose a random sample to replace:
                idx = np.random.randint(0, self.storage.shape[0])
                self.storage[idx] = row
        else:
            idx = self.total_samples - 1
            self.storage[idx] = row

    def get(self) -> np.ndarray:
        """
        Returns the logged data. This will be an array of shape
        (n_logged, 1 + dim), where the first column is the time,
        and the rest of the columns are the logged values at that time step.

        Returns:
            The logged data.
        """
        order = np.argsort(self.storage[: self.total_samples, 0])
        return self.storage[order]
