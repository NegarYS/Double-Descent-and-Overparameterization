import numpy as np
from tensor import Tensor
from function import *

class Module:
    def parameters(self):
        params = []
        for name, value in self.__dict__.items():
            if isinstance(value, Tensor) and value.requires_grad:
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
        return params

    def set_parameters(self, params):
        param_list = self.parameters()
        for i, p_data in enumerate(params):
            param_list[i].data = p_data
    
    def forward(self, *args, **kwargs): ...
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class Linear(Module):
    # TODO
    # Description: Initialize the Linear layer.
    # A linear layer performs the operation: output = input @ weights + bias.
    # This `__init__` method needs to create and initialize the layer's learnable
    # parameters: the weight matrix and the bias vector.
    #
    # 1. Weights: Create a `Tensor` for the weights.
    #    - Shape: (`in_features`, `out_features`). This allows for `input @ weights`.
    #    - Initialization: A good practice is to initialize weights with small random
    #      values to break symmetry. A common method is Kaiming He initialization,
    #      but for simplicity, you can use `np.random.randn(...) * np.sqrt(2 / in_features)`.
    #    - `requires_grad`: Must be `True`, as weights are learnable.
    #
    # 2. Bias: Create a `Tensor` for the bias.
    #    - Shape: (`out_features`,).
    #    - Initialization: Often initialized to zeros.
    #    - `requires_grad`: Must be `True`, as the bias is also learnable.
    #
    # Don't forget to call `super().__init__()`.
    def __init__(self, in_features, out_features):
        super().__init__()
        weight_data = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        bias_data = np.zeros((1, out_features), dtype=np.float32)

        self.weight = Tensor(weight_data, requires_grad=True)
        self.bias = Tensor(bias_data, requires_grad=True)

    # TODO
    # Description: Define the forward pass for the Linear layer.
    # This method applies the linear transformation to an input tensor `x`.
    #
    # The operation is `inputs @ weight + bias`.
    def forward(self, x):
        return MatMul.apply(x, self.weight) + self.bias

class ReLU(Module):
    # TODO
    # Description: Define the forward pass for the ReLU activation module.
    # This class acts as a modular wrapper for the ReLU `Function` you defined earlier.
    # Unlike `Linear`, ReLU has no learnable parameters, so it doesn't need an `__init__`
    # method to create weights or biases.
    #
    # The forward pass should simply apply the ReLU function to the input tensor `x`.
    #
    # Hint: You should call the `ReLU.apply(x)` function.
    def forward(self, x):
        return ReLU.apply(x)