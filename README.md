
# COCO - A Core Framework for Demystifying Deep Learning.  


<div align="center">
  <img src="Documentation/COCO - Cover.jpeg" alt="COCO Architecture" width="100%" />
</div>

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![License](https://img.shields.io/badge/Made_With-Python_&_NumPy-darkgreen.svg)]()
[![License](https://img.shields.io/badge/Purpose-Educational-darkred.svg)]()



## What is COCO?
> A lightweight deep learning framework focused on educational clarity and practical implementation. Developed from fundamental principles to demonstrate neural network mechanics.

## Features

### Core Components
- **Layer Architecture**: Flexible layer design with He/Xavier/Lecun initialization
- **Activation Functions**: ReLU, Leaky ReLU, Sigmoid, Tanh, Softmax
- **Optimization Algorithms**: 
  - SGD, Momentum, NAG
  - Adam, RMSprop, NAdam
- **Regularization Techniques**:
  - L1/L2 Regularization
  - Batch Normalization (1D & 2D)
  - Dropout

### Advanced Capabilities
- Automatic gradient computation
- Mini-batch training support
- Validation metrics tracking
- Training history visualization (Computational Graph + Feature Map Visualization)

## Installation

```bash
git clone https://github.com/WELOVECOCO/COCO.git
cd COCO
pip install -r requirements.txt
```

## Quick Start - Training Simple Neural Net

```python
import numpy as np
from core.Models import Model
from core.nn import Linear, Conv2d, MaxPool2d, batchnorm2d
from core.Functional import relu, softmax
from core.optim import Adam
from core.loss import sparse_categorical_cross_entropy
from core.tensor import Tensor

class NeuralNet(Model):
    def __init__(self):
        super().__init__()
        # Define network architecture
        self.conv1 = Conv2d(3, 16, kernel_size=3, activation=relu)
        self.pool1 = MaxPool2d(2)
        self.bn1 = batchnorm2d(16)
        
        self.conv2 = Conv2d(16, 32, kernel_size=3, activation=relu)
        self.pool2 = MaxPool2d(2)
        
        self.fc1 = Linear(32*6*6, 128, activation=relu)
        self.fc2 = Linear(128, 10, activation=softmax)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.bn1(x)
        
        x = self.conv2(x)
        x = self.pool2(x)
        
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = self.fc1(x)
        return self.fc2(x)

# Initialize model and optimizer
model = NeuralNet()
optimizer = Adam(model.trained_parameters(), learning_rate=0.001)
criterion = sparse_categorical_cross_entropy()

# Convert numpy data to tensors
X_train = Tensor(np.random.randn(100, 3, 32, 32))  # Example input
y_train = Tensor(np.random.randint(0, 10, (100, 10)))  # One-hot encoded

# Training loop
for epoch in range(10):
    # Forward pass
    outputs = model(X_train)
    loss = criterion.sparse_categorical_cross_entropy(y_train, outputs)
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print statistics
    print(f'Epoch {epoch+1}, Loss: {loss.data:.4f}')

# Evaluation
with model.test():  # Enable test mode (for batchnorm/dropout)
    X_test = Tensor(np.random.randn(10, 3, 32, 32))
    predictions = model(X_test).data
    print("Predictions:", predictions.argmax(axis=1))
```

## Documentation

### Layer Configuration
```python
Linear(
    input_dim,
    output_dim,
    initialize_type="random",
    activation="none",
    dropout=None,
    bias=True
)
```

### Available Optimizers
| Optimizer | Parameters       | Best For           |
|-----------|------------------|--------------------|
| SGD       | lr, momentum     | Simple networks    |
| Adam      | lr, beta1, beta2 | Most architectures |
| RMSprop   | lr, rho          | Unbalanced data    |
| NAdam     | lr, beta1, beta2 | Noisy gradients    |



## Benchmarks

TBD

## Contributing

We welcome contributions! Please follow these steps:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## License

Distributed under the MIT License. See `LICENSE` for more information.
