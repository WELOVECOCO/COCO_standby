import numpy as np
from core.Functional import *
from collections import deque
class Tensor:
    
    def get_topological_order(self, root):
        # iterative dfs with post order traversal (left->right->root)
        visited = set()
        post_order = []
        stack = [(root, False)]  # (node, has_been_processed)

        while stack:
            node, processed = stack.pop()

            if processed:
                post_order.append(node)
                continue

            if node in visited:
                continue

            visited.add(node)
            stack.append((node, True))  # Mark as processed

            for parent in node.parents:
                if parent not in visited:
                    stack.append((parent, False))

        self.order = post_order[::-1]  # Reverse post-order for topological order

    def __init__(self, data: np.ndarray, requires_grad: bool = True):
        self.data = data
        self.grad = None
        self.requires_grad = requires_grad
        self._grad_fn = None
        self.parents = []
        self.order = []

    
    def shape(self):
        return self.data.shape

    def assign_grad(self, grad):
        """Safely assign or accumulate gradients."""
        if not self.requires_grad:
            return
        
        if grad is None:
            raise ValueError("Gradient assigned to tensor is None")

        if self.grad is None:
            # print("self.grad is none")
            self.grad = grad
        else:
            # print("self.grad accum")
            self.grad += grad 

    def backward(self, grad=None,retain=False):
        if grad is None:
            if self.grad is None:
                grad = np.ones_like(self.data)
            else:
                grad = self.grad
            
            
        
        self.assign_grad(grad)
        
        if self.order == [] :
            
            self.get_topological_order(self)

        
        for node in self.order:
            if node._grad_fn is not None:
                node._grad_fn(node.grad)

        if retain==False:
            self.order = []

        

    def __add__(self, other):
        requires_grad = self.requires_grad or other.requires_grad
        
        out = Tensor(self.data + other.data, requires_grad)
        out._grad_fn = ADD_BACKWARD(self, other)  # class object TODO: maybe this should be a function
        out.parents = [self, other]
        return out
    
    def __mul__(self, other):
        requires_grad = self.requires_grad or other.requires_grad
        
        out = Tensor(self.data * other.data, requires_grad)
        out._grad_fn = MUL_BACKWARD(self, other)
        out.parents = [self, other]
        return out


    def __matmul__(self, other):
        requires_grad = self.requires_grad or other.requires_grad
        
        out_data = self.data @ other.data
        out = Tensor(out_data, requires_grad)
        

        out._grad_fn = MatMulBackward(self, other)
        out.parents = [self, other]
        return out
