import numpy as np
from core.tensor import Tensor

# Global constant for numerical stability in logarithms.
EPSILON = 1e-12

class binary_cross_entropy:
    def __init__(self):
        self.y_true = None
        self.y_pred = None
        self.loss = None
    def binary_cross_entropy(self,y_true, y_pred):
        if not isinstance(y_true, np.ndarray):
            try:
                y_true = y_true.toarray()
            except AttributeError:
                raise ValueError("y_true must be a numpy array or have a 'toarray' method.")

        if not isinstance(y_true, Tensor):
            y_true = Tensor(y_true, requires_grad=False)

        # Compute loss per sample
        self.y_true = y_true
        self.y_pred = y_pred
        y_pred_clipped = y_pred.clip(EPSILON, 1 - EPSILON)
        per_sample_loss = -(y_true * y_pred_clipped.log() + (1 - y_true) * (1 - y_pred_clipped).log())

        loss = per_sample_loss.mean()
        loss.parents = [y_pred]
        loss._grad_fn = self.grad_fn
        self.loss = loss


       
        return loss
    

    def grad_fn(self,grad):
        batch_size = self.y_pred.shape[0]
        self.loss.parents[0].assign_grad(grad * (self.y_pred.data - self.y_true.data) / batch_size)
        # print(self.loss.parents[0].grad)


class sparse_categorical_cross_entropy: #       
    def __init__(self):
        self.y_true = None
        self.y_pred = None
        self.loss = None

    
    def sparse_categorical_cross_entropy(self,y_true, y_pred, axis=1):
        if not isinstance(y_true, np.ndarray):
            try:
                y_true = y_true.toarray()
            except AttributeError:
                raise ValueError("y_true must be a numpy array or have a 'toarray' method.")

        if not isinstance(y_true, Tensor):
            y_true = Tensor(y_true, requires_grad=False)

        # Compute loss per sample
        self.y_true = y_true
        self.y_pred = y_pred
        per_sample_loss = -(y_true * (y_pred + EPSILON).log()).sum(axis=axis)

        loss = per_sample_loss.mean()
        loss.parents = [y_pred]
        loss._grad_fn = self.grad_fn
        self.loss = loss


       
        return loss
    

    def grad_fn(self,grad):
        batch_size = self.y_pred.shape[0]
        self.loss.parents[0].assign_grad(grad * (self.y_pred.data - self.y_true.data) / batch_size)
        # print(self.loss.parents[0].grad)

#    
class mean_squared_error:

    def __init__(self):
        self.y_true = None
        self.y_pred = None
        self.loss = None
    def mean_squared_error(self,y_true, y_pred):
        if not isinstance(y_true, np.ndarray):
            try:
                y_true = y_true.toarray()
            except AttributeError:
                raise ValueError("y_true must be a numpy array or have a 'toarray' method.")

        if not isinstance(y_true, Tensor):
            y_true = Tensor(y_true, requires_grad=False)

        # Compute loss per sample
        self.y_true = y_true
        self.y_pred = y_pred
        per_sample_loss = (y_true - y_pred) ** 2

        loss = per_sample_loss.mean()
        loss.parents = [y_pred]
        loss._grad_fn = self.grad_fn
        self.loss = loss


    
        return loss
    

    def grad_fn(self,grad):
        batch_size = self.y_pred.shape[0]
        dz = 2 * (self.y_pred.data - self.y_true.data)
        self.loss.parents[0].assign_grad(grad * dz / batch_size)



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
        return binary_cross_entropy()
    elif loss_fn_name == 'categorical_bce':
        return sparse_categorical_cross_entropy()
    elif loss_fn_name == 'mse':
        return mean_squared_error()
    else:
        raise ValueError("Unknown loss function: " + loss_fn_name)

