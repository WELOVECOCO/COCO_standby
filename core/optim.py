import numpy as np
from collections import defaultdict

def get_optimizer(name, learning_rate=0.001, beta1=0.9, beta2=0.999, EMA=False, clip_value=10):
    """
    Factory function to create an optimizer instance based on its name.

    Args:
        name (str): Name of the optimizer (e.g., "sgd", "momentum", "nag", "adam", "nadam", "adagrad", "rmsprop").
        learning_rate (float): Learning rate for the optimizer.
        beta1 (float): First momentum decay factor.
        beta2 (float): Second momentum decay factor (used by some optimizers).
        EMA (bool): Flag indicating whether to use Exponential Moving Average for momentum.
        clip_value (float): Maximum allowed absolute value for gradients (gradient clipping).

    Returns:
        Optimizer: An instance of the specified optimizer.

    Raises:
        ValueError: If the optimizer name is not recognized.
    """
    if name == "sgd":
        return sgd(learning_rate=learning_rate, beta1=beta1, beta2=beta2, EMA=EMA, clip_value=clip_value)
    elif name == "momentum":
        return momentum(learning_rate=learning_rate, beta1=beta1, beta2=beta2, EMA=EMA, clip_value=clip_value)
    elif name == "nag":
        return nag(learning_rate=learning_rate, beta1=beta1, beta2=beta2, EMA=EMA, clip_value=clip_value)
    elif name == "adam":
        return adam(learning_rate=learning_rate, beta1=beta1, beta2=beta2, EMA=EMA, clip_value=clip_value)
    elif name == "nadam":
        return NAdam(learning_rate=learning_rate, beta1=beta1, beta2=beta2, EMA=EMA, clip_value=clip_value)
    elif name == "adagrad":
        return AdaGrad(learning_rate=learning_rate, beta1=beta1, beta2=beta2, EMA=EMA, clip_value=clip_value)
    else:
        raise ValueError("Unknown optimizer: " + name)


class Optimizer:

    def __init__(self,params, learning_rate=0.001, beta1=0.9, beta2=0.999, EMA=False, clip_value=10):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.EMA = EMA
        self.clip_value = clip_value
        self.params = params


    def zero_grad(self):
        for param in self.params:
            param.zero_grad()
    def step(self):
        """
        Placeholder for the optimizer update logic.
        Subclasses must implement this method.

        Args:
            layer: Layer object containing parameters (weights, biases) and their gradients.
            **kwargs: Additional parameters for the update.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class sgd(Optimizer):
    def __init__(self,params, learning_rate=0.001, beta1=0.9, beta2=0.999, EMA=False, clip_value=10):
       super().__init__(params, learning_rate, beta1, beta2, EMA, clip_value)

    def step(self):
        for param in self.params:
            if param.grad is None:
                raise ValueError("Gradient for weights is None, cannot perform update.")
            param.data = param.data - self.learning_rate * param.grad


class momentum(Optimizer):
    """
    Momentum optimizer for gradient descent.
    """

    def __init__(self,params, learning_rate=0.001, beta1=0.9, beta2=0.999, EMA=False, clip_value=10):
       super().__init__(params, learning_rate, beta1, beta2, EMA, clip_value)
       self.momentum = defaultdict(lambda: None)

    def step(self):
        """
        Perform a momentum-based parameter update.

        Args:
            layer: Layer object with attributes:
                - weights, bias: Parameters.
                - grad_w, grad_b: Gradients.
                - momentum_w, momentum_b: Previous momentum values.
            **kwargs: Optional overrides for learning_rate, beta1, etc.
        """
        for param in self.params:

            if self.momentum[id(param)] is None:
                self.momentum[id(param)] = np.zeros_like(param.grad)

            if self.clip_value:
               param.grad = np.clip(param.grad, -self.clip_value, self.clip_value)

            if self.EMA:
                momentum_value = self.momentum[id(param)]
                momentum_value = self.beta1 * momentum_value + (1 - self.beta1) * param.grad
                self.momentum[id(param)] = momentum_value
                param.data -= self.learning_rate * momentum_value
            else:
                momentum_value = self.momentum[id(param)]
                momentum_value = self.beta1 * momentum_value + self.learning_rate * param.grad
                self.momentum[id(param)] = momentum_value
                param.data -= momentum_value


class nag(Optimizer):
    """
    Nesterov Accelerated Gradient (NAG) optimizer.
    """
    def __init__(self,params, learning_rate=0.001, beta1=0.9, beta2=0.999, EMA=False, clip_value=10):
       super().__init__(params, learning_rate, beta1, beta2, EMA, clip_value)
       self.momentum = defaultdict(lambda: None)

    def step(self):
        """
        Perform a momentum-based parameter update.

        Args:
            layer: Layer object with attributes:
                - weights, bias: Parameters.
                - grad_w, grad_b: Gradients.
                - momentum_w, momentum_b: Previous momentum values.
            **kwargs: Optional overrides for learning_rate, beta1, etc.
        """
        for param in self.params:

            if self.momentum[id(param)] is None:
                self.momentum[id(param)] = np.zeros_like(param.grad)

            if self.clip_value:
               param.grad = np.clip(param.grad, -self.clip_value, self.clip_value)

            if self.EMA:
                momentum_value = self.momentum[id(param)]
                momentum_value = self.beta1 * momentum_value + (1 - self.beta1) * param.grad
                self.momentum[id(param)] = momentum_value
                param.data -= (self.learning_rate * momentum_value)
            else:
                momentum_value = self.momentum[id(param)]
                momentum_value = self.beta1 * momentum_value + param.grad
                self.momentum[id(param)] = momentum_value
                param.data -= (momentum_value* self.learning_rate)

            # Lookahead step. TODO: Check this implementation for correctness
            param.data -= self.beta1 * self.momentum[id(param)]



class adam(Optimizer):
    """
    Adaptive Moment Estimation (Adam) optimizer.
    """

    def __init__(self,params, learning_rate=0.001, beta1=0.9, beta2=0.999, EMA=False, clip_value=10):
       super().__init__(params, learning_rate, beta1, beta2, EMA, clip_value)
       self.momentum = defaultdict(lambda: None)
       self.Gsquare = defaultdict(lambda: None)
       self.t = 0
       self.eps = 1e-8

    def step(self):
        self.t += 1
        for param in self.params:

            if self.momentum[id(param)] is None:
                self.momentum[id(param)] = np.zeros_like(param.grad)

            if self.Gsquare[id(param)] is None:
                self.Gsquare[id(param)] = np.zeros_like(param.grad)

            if self.clip_value:
               param.grad = np.clip(param.grad, -self.clip_value, self.clip_value)

            if self.EMA:
                momentum_value = self.momentum[id(param)]
                momentum_value = self.beta1 * momentum_value + (1 - self.beta1) * param.grad
                self.momentum[id(param)] = momentum_value
            else:
                momentum_value = self.momentum[id(param)]
                momentum_value = self.beta1 * momentum_value + self.learning_rate * param.grad
                self.momentum[id(param)] = momentum_value

            Gsquare_value = self.Gsquare[id(param)]
            # Update biased second raw moment estimate.
            Gsquare_value = self.beta2 * Gsquare_value + (1 - self.beta2) * (param.grad ** 2)
            self.Gsquare[id(param)] = Gsquare_value
            # Compute bias-corrected estimates.
            M_corrected = momentum_value / (1 - 0.99 ** self.t)
            G_corrected = Gsquare_value / (1 - 0.99 ** self.t)

            ita_w = self.learning_rate / (np.sqrt(G_corrected) + self.eps)

            param.data -= ita_w * M_corrected



class NAdam(Optimizer):
    """
    Nesterov-accelerated Adaptive Moment Estimation (NAdam) optimizer.
    """

    def __init__(self,params, learning_rate=0.001, beta1=0.9, beta2=0.999, EMA=False, clip_value=10):
       super().__init__(params, learning_rate, beta1, beta2, EMA, clip_value)
       self.momentum = defaultdict(lambda: None)
       self.Gsquare = defaultdict(lambda: None)
       self.t = 0

    def step(self):
        self.t += 1
        for param in self.params:

            if self.momentum[id(param)] is None:
                self.momentum[id(param)] = np.zeros_like(param.grad)

            if self.Gsquare[id(param)] is None:
                self.Gsquare[id(param)] = np.zeros_like(param.grad)

            if self.clip_value:
               param.grad = np.clip(param.grad, -self.clip_value, self.clip_value)

            if self.EMA:
                momentum_value = self.momentum[id(param)]
                momentum_value = self.beta1 * momentum_value + (1 - self.beta1) * param.grad
                self.momentum[id(param)] = momentum_value
            else:
                momentum_value = self.momentum[id(param)]
                momentum_value = self.beta1 * momentum_value + self.learning_rate * param.grad
                self.momentum[id(param)] = momentum_value

            Gsquare_value = self.Gsquare[id(param)]
            # Update biased second raw moment estimate.
            Gsquare_value = self.beta2 * Gsquare_value + (1 - self.beta2) * (param.grad ** 2)
            self.Gsquare[id(param)] = Gsquare_value
            # Compute bias-corrected estimates.
            M_corrected = momentum_value / (1 - 0.99 ** self.t)
            G_corrected = Gsquare_value / (1 - 0.99 ** self.t)

            ita_w = self.learning_rate / (np.sqrt(G_corrected) + self.eps)

            param.data -= ita_w * M_corrected

            param.data -= self.beta1 * self.momentum[id(param)]


class AdaGrad(Optimizer):
    """
    AdaGrad optimizer.
    """

    def __init__(self,params, learning_rate=0.001, beta1=0.9, beta2=0.999, EMA=False, clip_value=10):
       super().__init__(params, learning_rate, beta1, beta2, EMA, clip_value)
       self.Gsquare = defaultdict(lambda: None)
        # Note: AdaGrad typically does not use beta parameters.

    def __call__(self):
        for param in self.params:
            if self.Gsquare[id(param)] is None:
                self.Gsquare[id(param)] = np.zeros_like(param.grad)

            if self.clip_value:
               param.grad = np.clip(param.grad, -self.clip_value, self.clip_value)

            Gsquare_value = self.Gsquare[id(param)]
            # Update biased second raw moment estimate.
            Gsquare_value = self.beta2 * Gsquare_value + (1 - self.beta2) * (param.grad ** 2)
            self.Gsquare[id(param)] = Gsquare_value

        ita_w = self.learning_rate / (np.sqrt(Gsquare_value) + self.eps)

        param.data -= ita_w * param.grad



