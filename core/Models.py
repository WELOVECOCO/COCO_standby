import h5py
from core.nn import *

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
        if isinstance(value, (Layer, Module, Activation, Model)):
            self.layers[name] = value

    def parameters(self):
        """
        Return the model parameters. If a layer is a composite model, recursively collect its parameters.
        """
        params = []
        for layer in self.layers.values():
            if isinstance(layer, Model):  # Composite model
                params += layer.parameters()
            elif isinstance(layer, Layer):
                params += layer.parameters()
            # You can add additional cases if other types hold parameters.
        return params

    def forward(self, x):
        """
        Users must override this method to define the forward pass.
        """
        raise NotImplementedError("Forward method must be implemented in subclasses")

    def __call__(self, x):
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

    def save_model(self, filepath):
        """
        Save the model parameters to an .h5 file.
        For composite models, save the parameters of each sub-model recursively.
        """
        with h5py.File(filepath, 'w') as f:
            for name, layer in self.layers.items():
                # If the layer is a composite model, create a group for it and let it save its parameters.
                if isinstance(layer, Model):
                    grp = f.create_group(name)
                    layer._save_to_group(grp)
                else:
                    if hasattr(layer, 'weights'):
                        f.create_dataset(f'{name}/weights', data=layer.weights.data)
                    if hasattr(layer, 'bias'):
                        f.create_dataset(f'{name}/bias', data=layer.bias.data)

    def _save_to_group(self, group):
        """
        Helper function to save parameters into an existing HDF5 group.
        """
        for name, layer in self.layers.items():
            if isinstance(layer, Model):
                grp = group.create_group(name)
                layer._save_to_group(grp)
            else:
                if hasattr(layer, 'weights'):
                    group.create_dataset(f'{name}/weights', data=layer.weights.data)
                if hasattr(layer, 'bias'):
                    group.create_dataset(f'{name}/bias', data=layer.bias.data)

    def load_model(self, filepath):
        """
        Load model parameters from an .h5 file.
        The architecture of the model (including composite models) must match the saved file.
        """
        with h5py.File(filepath, 'r') as f:
            for name, layer in self.layers.items():
                if isinstance(layer, Model):
                    if name in f:
                        layer._load_from_group(f[name])
                else:
                    if hasattr(layer, 'weights') and f.get(f'{name}/weights'):
                        layer.weights.data = f[f'{name}/weights'][:]
                    if hasattr(layer, 'bias') and f.get(f'{name}/bias'):
                        layer.bias.data = f[f'{name}/bias'][:]

    def _load_from_group(self, group):
        """
        Helper function to load parameters from an existing HDF5 group.
        """
        for name, layer in self.layers.items():
            if isinstance(layer, Model):
                if name in group:
                    layer._load_from_group(group[name])
            else:
                if hasattr(layer, 'weights') and group.get(f'{name}/weights'):
                    layer.weights.data = group[f'{name}/weights'][:]
                if hasattr(layer, 'bias') and group.get(f'{name}/bias'):
                    layer.bias.data = group[f'{name}/bias'][:]
