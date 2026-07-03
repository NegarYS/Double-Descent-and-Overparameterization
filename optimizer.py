from tensor import Tensor
from typing import List
import numpy as np

# TODO
# Description: Implements the Stochastic Gradient Descent (SGD) optimizer.
# Optimizers are responsible for updating a model's parameters (weights and biases)
# based on the gradients computed during backpropagation. The goal is to minimize the
# loss function by iteratively adjusting the parameters in the opposite direction
# of their gradients.
#
# __init__:
# The constructor takes a list of the model's parameters (`params`) that it will
# manage and a learning rate (`lr`). The learning rate is a crucial hyperparameter
# that controls the size of the update steps.
class SGD:
    def __init__(self, params: List[Tensor], lr: float = 0.01, momentum: float = 0.0):
        if not isinstance(params, list) or not all(isinstance(p, Tensor) for p in params):
            raise TypeError("`params` must be a list of `Tensor` objects")

        self.params = params
        self.lr = lr
        self.momentum = momentum

        self.velocities = {}
        if self.momentum != 0.0:
            for p in self.params:
                self.velocities[p] = np.zeros_like(p.data)

    # TODO
    # Description: Performs a single optimization step.
    # This method iterates through all the parameters registered with the optimizer and
    # updates their values. The update rule for standard SGD is:
    # `parameter.data = parameter.data - learning_rate * parameter.grad`
    #
    # Important: This operation should directly modify the `.data` attribute of each
    # parameter Tensor. It is an in-place update that should NOT be tracked by the
    # autograd engine (i.e., it should not create new nodes in the computation graph).
    def step(self):

        for p in self.params:
            if p.requires_grad and p.grad is not None:

                if self.momentum != 0.0:
                    v_prev = self.velocities[p]
                    v_new = self.momentum * v_prev - self.lr * p.grad                    
                    self.velocities[p] = v_new                    
                    p.data += v_new

                else:
                    p.data -= self.lr * p.grad



    # TODO
    # Description: Resets the gradients of all registered parameters to zero.
    # Because gradients are accumulated (summed up) in the `.grad` attribute during
    # the backward pass, you must clear them before starting a new training iteration.
    # If you forget to do this, the gradients from the new batch will be added to the
    # gradients from the previous batch, leading to an incorrect optimization step.
    #
    # This method should be called at the beginning of each training loop, typically
    # right before the forward pass.
    def zero_grad(self):
        for param in self.params:
            if param.requires_grad:
                param.grad = np.zeros_like(param.grad)


# Adam Optimizer
class Adam:
    def __init__(self, params: List[Tensor], lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        if not isinstance(params, list) or not all(isinstance(p, Tensor) for p in params):
            raise TypeError("`params` must be a list of `Tensor` objects")

        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in params]  
        self.v = [np.zeros_like(p.data) for p in params]  
        self.t = 0  

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            if param.requires_grad:
               
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (param.grad ** 2)
                
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                
                param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for param in self.params:
            if param.requires_grad:
                param.grad = np.zeros_like(param.grad)
