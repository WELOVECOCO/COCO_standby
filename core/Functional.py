
class ADD_BACKWARD:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, grad):
        if self.a.requires_grad:
            self.a.assign_grad(grad)

        if self.b.requires_grad:
            self.b.assign_grad(grad)



class MUL_BACKWARD:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, grad):
        if grad is None:
            raise ValueError("Gradient is None")
        if self.a.requires_grad:
            self.a.assign_grad(grad * self.b.data)

        if self.b.requires_grad:
            self.b.assign_grad(grad * self.a.data)


class MatMulBackward:
    def __init__(self, A, B):
        self.A = A
        self.B = B

    def __call__(self, grad):
        # For A: grad_A = grad @ B.T
        if self.A.requires_grad:
            self.A.assign_grad(grad @ self.B.data.T)
        # For B: grad_B = A.T @ grad
        if self.B.requires_grad:
            self.B.assign_grad(self.A.data.T @ grad)
