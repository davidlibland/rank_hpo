import numpy as np


class TimeSeriesLogger:
    def __init__(self, max_samples, dim=1):
        self.max_samples = max_samples
        self.total_samples = 0
        self.dim = dim
        self.storage = np.full((max_samples, 1 + dim), np.nan)
        # The first column stores the time, the second column stores the value.

    def log(self, x):
        """Logs a new value, subsampling if necessary."""
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

    def get(self):
        """Returns the logged data."""
        order = np.argsort(self.storage[: self.total_samples, 0])
        return self.storage[order]
