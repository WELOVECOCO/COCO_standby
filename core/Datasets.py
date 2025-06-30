import numpy as np
from core.new_tensor import Tensor


class Dataset:
    def __init__(self, X, y, batch_size, shuffle=True, num_classes=10):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_classes = num_classes
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

        if hasattr(X_batch, "toarray"):
            X_batch = X_batch.toarray()
        if hasattr(y_batch, "toarray"):
            y_batch = y_batch.toarray()

        # One-hot encode labels
        y_onehot = np.eye(self.num_classes)[y_batch]

        return Tensor(X_batch, requires_grad=True), Tensor(y_onehot, requires_grad=False)

    def reset(self):
        self.current = 0
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, index):
        if index < 0 or index >= len(self.indices):
            raise IndexError("Index out of bounds")
        batch_indices = self.indices[index:index + self.batch_size]
        X_batch = self.X[batch_indices]
        y_batch = self.y[batch_indices]

        if hasattr(X_batch, "toarray"):
            X_batch = X_batch.toarray()
        if hasattr(y_batch, "toarray"):
            y_batch = y_batch.toarray()

        # One-hot encode labels
        y_onehot = np.eye(self.num_classes)[y_batch]

        return Tensor(X_batch, requires_grad=True), Tensor(y_onehot, requires_grad=False)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.reset()




@staticmethod
def mnist(batch_size=64, shuffle=True):
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split

    # Load dataset
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist['data'].astype(np.float32)
    y = mnist['target'].astype(np.int64)

    # Normalize and reshape
    X = X / 255.0
    X = X.reshape(-1, 1, 28, 28)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Pass num_classes=10 to enable one-hot encoding
    train_dataset = Dataset(X_train, y_train, batch_size=batch_size, shuffle=shuffle, num_classes=10)
    test_dataset = Dataset(X_test, y_test, batch_size=batch_size, shuffle=False, num_classes=10)

    return train_dataset, test_dataset

