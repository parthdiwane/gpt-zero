import numpy as np


class Tensor:
    """
        data: numpy array
        requires_grad: boolean 
        grad: gradient object for this tensor
        _node: node that contains operations (add, sub, multi, etc) of parents
    """
    def __init__(self, data, requires_grad, grad = None, _node = None):
        self.data = data 
        self.grad = grad # numpy array
        self.requires_grad = requires_grad
        self._node = _node    

    @property
    def shape(self):
        return self.data.shape
    @property
    def dtype(self):
        return self.data.dtype

    def __repr__(self):
        return "Tensor(shape=(" + str(self.shape) + ",requires_grad=" + str(self.requires_grad) + ")"
    





class backprop:
    def __init__(self):
        pass

