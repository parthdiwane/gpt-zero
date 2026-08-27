"""
genrerated by gen_derivatives.py. look at derivatives.yaml to change stuff
"""
import numpy as np
from derivative_helpers import *


def add_forward(self, other):
    result = self + other
    cache = {'self': self, 'other': other, 'result': result}
    return result, cache

def add_backward(grad, cache):
    self, other, result = cache['self'], cache['other'], cache['result']
    d_self = unbroadcast(grad, np.shape(self))
    d_other = unbroadcast(grad, np.shape(other))
    return (d_self, d_other)

def sub_forward(self, other):
    result = self - other
    cache = {'self': self, 'other': other, 'result': result}
    return result, cache

def sub_backward(grad, cache):
    self, other, result = cache['self'], cache['other'], cache['result']
    d_self = unbroadcast(grad, np.shape(self))
    d_other = unbroadcast(-grad, np.shape(other))
    return (d_self, d_other)

def mul_forward(self, other):
    result = self * other
    cache = {'self': self, 'other': other, 'result': result}
    return result, cache

def mul_backward(grad, cache):
    self, other, result = cache['self'], cache['other'], cache['result']
    d_self = unbroadcast(grad * other, np.shape(self))
    d_other = unbroadcast(grad * self, np.shape(other))
    return (d_self, d_other)

def div_forward(self, other):
    result = self / other
    cache = {'self': self, 'other': other, 'result': result}
    return result, cache

def div_backward(grad, cache):
    self, other, result = cache['self'], cache['other'], cache['result']
    d_self = unbroadcast(grad / other, np.shape(self))
    d_other = unbroadcast(-grad * self / (other * other), np.shape(other))
    return (d_self, d_other)

def neg_forward(self):
    result = -self
    cache = {'self': self, 'result': result}
    return result, cache

def neg_backward(grad, cache):
    self, result = cache['self'], cache['result']
    d_self = unbroadcast(-grad, np.shape(self))
    return (d_self,)

def pow_forward(self, exponent):
    result = self ** exponent
    cache = {'self': self, 'exponent': exponent, 'result': result}
    return result, cache

def pow_backward(grad, cache):
    self, exponent, result = cache['self'], cache['exponent'], cache['result']
    d_self = unbroadcast(grad * exponent * self ** (exponent - 1), np.shape(self))
    return (d_self,)

def exp_forward(self):
    result = np.exp(self)
    cache = {'self': self, 'result': result}
    return result, cache

def exp_backward(grad, cache):
    self, result = cache['self'], cache['result']
    d_self = unbroadcast(grad * result, np.shape(self))
    return (d_self,)

def log_forward(self):
    result = np.log(self)
    cache = {'self': self, 'result': result}
    return result, cache

def log_backward(grad, cache):
    self, result = cache['self'], cache['result']
    d_self = unbroadcast(grad / self, np.shape(self))
    return (d_self,)

def sqrt_forward(self):
    result = np.sqrt(self)
    cache = {'self': self, 'result': result}
    return result, cache

def sqrt_backward(grad, cache):
    self, result = cache['self'], cache['result']
    d_self = unbroadcast(grad / (2.0 * result), np.shape(self))
    return (d_self,)

def tanh_forward(self):
    result = np.tanh(self)
    cache = {'self': self, 'result': result}
    return result, cache

def tanh_backward(grad, cache):
    self, result = cache['self'], cache['result']
    d_self = unbroadcast(grad * (1.0 - result * result), np.shape(self))
    return (d_self,)

def sigmoid_forward(self):
    result = 1.0 / (1.0 + np.exp(-self))
    cache = {'self': self, 'result': result}
    return result, cache

def sigmoid_backward(grad, cache):
    self, result = cache['self'], cache['result']
    d_self = unbroadcast(grad * result * (1.0 - result), np.shape(self))
    return (d_self,)

def relu_forward(self):
    result = np.maximum(self, 0.0)
    cache = {'self': self, 'result': result}
    return result, cache

def relu_backward(grad, cache):
    self, result = cache['self'], cache['result']
    d_self = unbroadcast(grad * (self > 0.0), np.shape(self))
    return (d_self,)

def gelu_forward(self):
    result = gelu_fn(self)
    cache = {'self': self, 'result': result}
    return result, cache

def gelu_backward(grad, cache):
    self, result = cache['self'], cache['result']
    d_self = unbroadcast(gelu_grad(grad, self), np.shape(self))
    return (d_self,)

def matmul_forward(self, other):
    result = self @ other
    cache = {'self': self, 'other': other, 'result': result}
    return result, cache

def matmul_backward(grad, cache):
    self, other, result = cache['self'], cache['other'], cache['result']
    d_self = unbroadcast(grad @ np.swapaxes(other, -1, -2), np.shape(self))
    d_other = unbroadcast(np.swapaxes(self, -1, -2) @ grad, np.shape(other))
    return (d_self, d_other)

def sum_forward(self, dim=None, keepdim=False):
    result = np.sum(self, axis=dim, keepdims=keepdim)
    cache = {'self': self, 'dim': dim, 'keepdim': keepdim, 'result': result}
    return result, cache

def sum_backward(grad, cache):
    self, dim, keepdim, result = cache['self'], cache['dim'], cache['keepdim'], cache['result']
    d_self = unbroadcast(unreduce(grad, np.shape(self), dim, keepdim), np.shape(self))
    return (d_self,)

def mean_forward(self, dim=None, keepdim=False):
    result = np.mean(self, axis=dim, keepdims=keepdim)
    cache = {'self': self, 'dim': dim, 'keepdim': keepdim, 'result': result}
    return result, cache

def mean_backward(grad, cache):
    self, dim, keepdim, result = cache['self'], cache['dim'], cache['keepdim'], cache['result']
    d_self = unbroadcast(unreduce(grad, np.shape(self), dim, keepdim) / reduced_count(np.shape(self), dim), np.shape(self))
    return (d_self,)

def max_forward(self, dim=None, keepdim=False):
    result = np.max(self, axis=dim, keepdims=keepdim)
    cache = {'self': self, 'dim': dim, 'keepdim': keepdim, 'result': result}
    return result, cache

def max_backward(grad, cache):
    self, dim, keepdim, result = cache['self'], cache['dim'], cache['keepdim'], cache['result']
    d_self = unbroadcast(unreduce(grad, np.shape(self), dim, keepdim) * (self == unreduce(result, np.shape(self), dim, keepdim)), np.shape(self))
    return (d_self,)

def reshape_forward(self, shape):
    result = np.reshape(self, shape)
    cache = {'self': self, 'shape': shape, 'result': result}
    return result, cache

def reshape_backward(grad, cache):
    self, shape, result = cache['self'], cache['shape'], cache['result']
    d_self = unbroadcast(np.reshape(grad, np.shape(self)), np.shape(self))
    return (d_self,)

def transpose_forward(self, dim0=-2, dim1=-1):
    result = np.swapaxes(self, dim0, dim1)
    cache = {'self': self, 'dim0': dim0, 'dim1': dim1, 'result': result}
    return result, cache

def transpose_backward(grad, cache):
    self, dim0, dim1, result = cache['self'], cache['dim0'], cache['dim1'], cache['result']
    d_self = unbroadcast(np.swapaxes(grad, dim0, dim1), np.shape(self))
    return (d_self,)

def permute_forward(self, dims):
    result = np.transpose(self, dims)
    cache = {'self': self, 'dims': dims, 'result': result}
    return result, cache

def permute_backward(grad, cache):
    self, dims, result = cache['self'], cache['dims'], cache['result']
    d_self = unbroadcast(permute_grad(grad, dims), np.shape(self))
    return (d_self,)

def embedding_forward(weight, index):
    result = weight[index]
    cache = {'weight': weight, 'index': index, 'result': result}
    return result, cache

def embedding_backward(grad, cache):
    weight, index, result = cache['weight'], cache['index'], cache['result']
    d_weight = unbroadcast(scatter_add(np.shape(weight), index, grad, weight.dtype), np.shape(weight))
    return (d_weight,)

def softmax_forward(self, dim=-1):
    result = np.exp(self - np.max(self, axis=dim, keepdims=True)) / np.sum(np.exp(self - np.max(self, axis=dim, keepdims=True)), axis=dim, keepdims=True)
    cache = {'self': self, 'dim': dim, 'result': result}
    return result, cache

def softmax_backward(grad, cache):
    self, dim, result = cache['self'], cache['dim'], cache['result']
    d_self = unbroadcast((grad - np.sum(grad * result, axis=dim, keepdims=True)) * result, np.shape(self))
    return (d_self,)

def log_softmax_forward(self, dim=-1):
    result = log_softmax_stable(self, dim)
    cache = {'self': self, 'dim': dim, 'result': result}
    return result, cache

def log_softmax_backward(grad, cache):
    self, dim, result = cache['self'], cache['dim'], cache['result']
    d_self = unbroadcast(grad - np.exp(result) * np.sum(grad, axis=dim, keepdims=True), np.shape(self))
    return (d_self,)

def masked_fill_forward(self, mask, value):
    result = np.where(mask, value, self)
    cache = {'self': self, 'mask': mask, 'value': value, 'result': result}
    return result, cache

def masked_fill_backward(grad, cache):
    self, mask, value, result = cache['self'], cache['mask'], cache['value'], cache['result']
    d_self = unbroadcast(grad * np.logical_not(mask), np.shape(self))
    return (d_self,)

def layernorm_forward(self, weight, bias, eps=1e-05):
    result = layernorm_apply(self, weight, bias, eps)
    cache = {'self': self, 'weight': weight, 'bias': bias, 'eps': eps, 'result': result}
    return result, cache

def layernorm_backward(grad, cache):
    self, weight, bias, eps, result = cache['self'], cache['weight'], cache['bias'], cache['eps'], cache['result']
    d_self = unbroadcast(layernorm_grad_input(grad, self, weight, eps), np.shape(self))
    d_weight = unbroadcast(grad * layernorm_xhat(self, eps), np.shape(weight))
    d_bias = unbroadcast(grad, np.shape(bias))
    return (d_self, d_weight, d_bias)

def cross_entropy_forward(self, target):
    result = cross_entropy_loss(self, target)
    cache = {'self': self, 'target': target, 'result': result}
    return result, cache

def cross_entropy_backward(grad, cache):
    self, target, result = cache['self'], cache['target'], cache['result']
    d_self = unbroadcast(cross_entropy_grad_logits(grad, self, target), np.shape(self))
    return (d_self,)

FORWARD = {'add': add_forward, 'sub': sub_forward, 'mul': mul_forward, 'div': div_forward, 'neg': neg_forward, 'pow': pow_forward, 'exp': exp_forward, 'log': log_forward, 'sqrt': sqrt_forward, 'tanh': tanh_forward, 'sigmoid': sigmoid_forward, 'relu': relu_forward, 'gelu': gelu_forward, 'matmul': matmul_forward, 'sum': sum_forward, 'mean': mean_forward, 'max': max_forward, 'reshape': reshape_forward, 'transpose': transpose_forward, 'permute': permute_forward, 'embedding': embedding_forward, 'softmax': softmax_forward, 'log_softmax': log_softmax_forward, 'masked_fill': masked_fill_forward, 'layernorm': layernorm_forward, 'cross_entropy': cross_entropy_forward}

VJP = {'add': add_backward, 'sub': sub_backward, 'mul': mul_backward, 'div': div_backward, 'neg': neg_backward, 'pow': pow_backward, 'exp': exp_backward, 'log': log_backward, 'sqrt': sqrt_backward, 'tanh': tanh_backward, 'sigmoid': sigmoid_backward, 'relu': relu_backward, 'gelu': gelu_backward, 'matmul': matmul_backward, 'sum': sum_backward, 'mean': mean_backward, 'max': max_backward, 'reshape': reshape_backward, 'transpose': transpose_backward, 'permute': permute_backward, 'embedding': embedding_backward, 'softmax': softmax_backward, 'log_softmax': log_softmax_backward, 'masked_fill': masked_fill_backward, 'layernorm': layernorm_backward, 'cross_entropy': cross_entropy_backward}