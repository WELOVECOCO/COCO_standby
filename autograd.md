how do we have a ready autograd?

what we have now :
    - Tensor class
        -data
        -requires grad
        -grad
        -grad_fn (operation that created that tensor)
        -parents (refrence to the Tensors contributed in creating that tensor)
        -elementry operations and their backward
        -backward function to calculate the gradients for all the graph  starting from this point (dfs on all Tensors - reverse the dfs order - for each tensor : execute its grad_fn)
        -add a functionality to freeze the graph and reuse it if the user asked so that it is calculated only once (like a static graph)
        - adapt the layers to the auto grad (how ?)
            - the forward function returns a tensor whos _grad_fn is the layer backward function
            - the backward function does not return anything now in _grad_fn (called by the output tensor of the layer) we just update (self.weight.grad-self.bias.grad-self.input.grad)
        -fuse layers with activation functions to reduce number of nodes in a computational graph
    -projection in residual connections y = F(x) + x  when x and F(x) are not the same size (we perform a 1x1 convolution)
    - more elementry operations on the tensor (reshaping - transpose - alot more)
    -adapt the optimizers (they take a list containing all the weights of the network and they loop on it and update it)
    -also optimizer has a state dict containing the momentum and acc_grad_square values for each single weight and bias

    
what we need :