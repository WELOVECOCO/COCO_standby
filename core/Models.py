import h5py
from core.nn import *
from core.config import Config
class Model:
    def __init__(self):
        """
        Initialize the model.

        Attributes:
            layers (dict): Dictionary to store layers (or sub-models) with their variable names.
            losses (list): Store training losses.
            last_out (ndarray or None): Store the last output for loss computation.
        """
        
        self.layers = {}   # Dictionary to store layers/sub-models with their variable names
        self.losses = []   # Store training losses
        self.last_out = None  # Store the last output for loss computation

    def __setattr__(self, name, value):
        """
        Override __setattr__ to track layers and sub-models.
        """
        super().__setattr__(name, value)
        # Include Model instances along with Layer, Module, and Activation.
        if isinstance(value, (Layer, Module, Model)):
            self.layers[name] = value


    def to(self, device):
        """
        Move the model to the specified device (e.g., CPU or GPU).
        """
        for layer in self.layers.values():
            if isinstance(layer, Model):
                layer.to(device)
            elif hasattr(layer, 'to'):
                layer.to(device)
        return self

    def parameters(self):
        params = []
        seen = set()  # Track seen objects to avoid duplicates
        for layer in self.layers.values():
            
            if id(layer) in seen:
                continue
            seen.add(id(layer))
            if isinstance(layer, Model):
                params += layer.parameters()
            elif isinstance(layer, Layer):
                params += layer.parameters()
        return params

    
    def trained_parameters(self):
        params = []
        seen = set()  # Track seen objects to avoid duplicates
        for layer in self.layers.values():
            if id(layer) in seen:
                continue
            seen.add(id(layer))
            if isinstance(layer, Model):
                params += layer.trained_parameters()
            elif isinstance(layer, Layer):
                params += layer.trained_parameters()
        return params


    def test(self):
        Config.TEST = True
    def forward(self, x, test=False):
        """
        Users must override this method to define the forward pass.
        """
        raise NotImplementedError("Forward method must be implemented in subclasses")

    def __call__(self, x, test=False):
        """
        ARGS: 
            x (ndarray): Input data (e.g., image: [B, C, H, W] or feature vector: [B, D])
        RETURNS:
            ndarray: Output probabilities (B, num_output neurons)
        """
        self.last_out = self.forward(x)
        return self.last_out

    def summary(self, indent=0):
        """
        Print a summary of the model architecture. For composite models, recursively print sub-model details.
        """
        prefix = " " * indent
        print(prefix + "Model Summary:")
        print(prefix + "=" * 50)
        for name, layer in self.layers.items():
            if isinstance(layer, Model):
                print(prefix + f"{name}: Composite Model")
                layer.summary(indent=indent + 4)
            else:
                print(prefix + f"{name}: {layer.__class__.__name__}")
                if hasattr(layer, 'weights'):
                    print(prefix + f"  Weights: {layer.weights.shape}")
                if hasattr(layer, 'bias'):
                    print(prefix + f"  Bias: {layer.bias.shape}")
        print(prefix + "=" * 50)


    def view_graph(self, input_data, filename="model_graph", view=True):
        """
        Runs a forward pass using the provided input_data and visualizes the
        computation graph of the output tensor.
        """
        output_tensor = self(Tensor(input_data,requires_grad=True))
        # Assuming output_tensor is an instance of Tensor and has view_graph
        output_tensor.view_graph(filename=filename,view=view)

    def load_weights_by_structure(self, state_dict, strict=True):

        # Get the model's parameters in order
        model_params = self.trained_parameters()  # Assumes parameters() returns a list of Tensors

        # Flatten the state_dict values into a list (ignore keys)
        pretrained_weights = list(state_dict.values())
        
        if len(pretrained_weights) != len(model_params):
            warning = (f"Number of pre-trained weights ({len(pretrained_weights)}) does not match "
                    f"number of model parameters ({len(model_params)})")
            if strict:
                raise ValueError(warning)
            else:
                print(f"Warning: {warning}")

        # Assign weights based on order and check shapes
        for i, (param, weight) in enumerate(zip(model_params, pretrained_weights)):
            weight = np.array(weight)
            if weight.ndim == 2:
                weight = weight.T
            if param.data.ndim == 4 and weight.ndim == 1 and weight.shape[0] == param.data.shape[1]:
                weight = weight.reshape((1, weight.shape[0], 1, 1))
            if param.data.shape != weight.shape:
                error_msg = (f"Shape mismatch at parameter {i}: "
                            f"model expects {param.data.shape}, "
                            f"pre-trained weight has {weight.shape}")
                if strict:
                    raise ValueError(error_msg)
                else:
                    print(f"Warning: {error_msg}")
                    continue
            param.data = weight

        if not strict and len(pretrained_weights) > len(model_params):
            print(f"Warning: {len(pretrained_weights) - len(model_params)} pre-trained weights were unused")