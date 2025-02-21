# augmentation/datasets/dataset.py
import numpy as np
from core.tensor import Tensor


class Dataset:
    def __init__(self, X, y, batch_size, transform=None, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.transform = transform
        self.shuffle = shuffle
        self.indices = np.arange(X.shape[0])
        if shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        if self.current >= len(self.indices):
            raise StopIteration
        batch_indices = self.indices[self.current:self.current + self.batch_size]
        self.current += self.batch_size
        X_batch = self.X[batch_indices]
        y_batch = self.y[batch_indices]

        # Convert from sparse if needed.
        if hasattr(X_batch, "toarray"):
            X_batch = X_batch.toarray()
        if hasattr(y_batch, "toarray"):
            y_batch = y_batch.toarray()

        if self.transform:
            X_batch = self.transform(X_batch)

        return X_batch, y_batch

    def reset(self):
        self.current = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
