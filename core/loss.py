import numpy as np
from core.tensor import Tensor

# Global constant for numerical stability in logarithms.
EPSILON = 1e-12


def binary_cross_entropy(y_true, y_pred):
    """
    Compute the binary cross entropy loss and its gradient.

    Assumes:
        - y_true is an array of shape (B, 1), where B is the batch size.
        - y_pred is an array of shape (B, 1) containing predicted probabilities,
          with values in (0, 1).

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground truth labels.
    y_pred : numpy.ndarray
        Predicted probabilities.

    Returns
    -------
    loss : numpy.ndarray
        The average binary cross entropy loss (averaged over the batch).
    grad : numpy.ndarray
        The gradient of the loss with respect to y_pred.

    Notes
    -----
    A small constant EPSILON is added to y_pred (and 1 - y_pred) to avoid log(0) issues.
    """
    # Convert y_true to a Tensor (no gradient needed)
    if not isinstance(y_true, Tensor):
        y_true = Tensor(y_true, requires_grad=False)
    # Use Tensor operations to build the computation graph.
    y_pred_clipped = y_pred.clip(EPSILON, 1 - EPSILON)
    loss = -(y_true * y_pred_clipped.log() + (Tensor(1.0, requires_grad=False) - y_true) *
             (Tensor(1.0, requires_grad=False) - y_pred_clipped).log()).mean()
    return loss


def sparse_categorical_cross_entropy(y_true, y_pred, axis=1):
    """
    Compute the sparse categorical cross entropy loss and its gradient.

    This function converts a sparse matrix (if applicable) to dense format before
    calculating the loss.

    Assumes:
        - y_true is either a dense numpy array or a sparse matrix (with a .toarray() method)
          of shape (B, num_classes), where B is the batch size.
        - y_pred is a numpy array of shape (B, num_classes) containing probability
          distributions along the specified axis.

    Parameters
    ----------
    y_true : numpy.ndarray or sparse matrix
        Ground truth labels. If not a numpy array, it is converted using .toarray().
    y_pred : numpy.ndarray
        Predicted probability distributions.
    axis : int, optional
        The axis along which the classes are defined (default is 1).

    Returns
    -------
    loss : numpy.ndarray
        The average sparse categorical cross entropy loss.
    grad : numpy.ndarray
        The gradient of the loss with respect to y_pred.

    Notes
    -----
    A small constant EPSILON is added to y_pred before taking the logarithm to avoid log(0).
    """
    # Convert sparse matrix to dense if needed.
    if not isinstance(y_true, np.ndarray):
        try:
            y_true = y_true.toarray()
        except AttributeError:
            raise ValueError("y_true must be a numpy array or have a 'toarray' method.")
    if not isinstance(y_true, Tensor):
        y_true = Tensor(y_true, requires_grad=False)
    # Build the loss using Tensor operations.
    loss = -(y_true * (y_pred + EPSILON).log()).sum(axis=axis).mean()
    return loss


def mean_squared_error(y_true, y_pred):
    """
    Compute the mean squared error (MSE) loss and its gradient.

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground truth values.
    y_pred : numpy.ndarray
        Predicted values.

    Returns
    -------
    loss : numpy.ndarray
        The average mean squared error loss.
    grad : numpy.ndarray
        The gradient of the MSE loss with respect to y_pred.

    Notes
    -----
    The gradient here is computed as 2*(y_pred - y_true)/B, where B is the batch size.
    """
    if not isinstance(y_true, Tensor):
        y_true = Tensor(y_true, requires_grad=False)
    loss = ((y_true - y_pred) ** 2).mean()
    return loss


def get_loss_fn(loss_fn_name):
    """
    Retrieve the loss function based on the provided name.

    Parameters
    ----------
    loss_fn_name : str
        Name of the loss function. Options are:
            - 'bce' for binary cross entropy
            - 'categorical_bce' for sparse categorical cross entropy
            - 'mse' for mean squared error

    Returns
    -------
    function
        The corresponding loss function.

    Raises
    ------
    ValueError
        If the loss function name is not recognized.
    """
    if loss_fn_name == 'bce':
        return binary_cross_entropy
    elif loss_fn_name == 'categorical_bce':
        return sparse_categorical_cross_entropy
    elif loss_fn_name == 'mse':
        return mean_squared_error
    else:
        raise ValueError("Unknown loss function: " + loss_fn_name)


def loss_wrapper(loss_fn, y_true, y_pred):
    """
    Wrap a loss function so that it returns a Tensor with a custom backward function.

    Parameters:
      loss_fn: a function that takes (y_true, y_pred_data) and returns (loss_value, grad)
      y_true: the ground-truth labels (numpy array)
      y_pred: the prediction Tensor (our custom Tensor) from the network.

    Returns:
      A Tensor that holds the loss value and whose backward function propagates the loss gradient to y_pred.
    """
    # Compute the loss using the provided loss function, which now returns a Tensor.
    loss_tensor = loss_fn(y_true, y_pred)
    return loss_tensor
