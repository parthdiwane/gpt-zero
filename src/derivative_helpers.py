"""Runtime helpers referenced by the formulas in derivatives.yaml.

The generated code in _derivatives_generated.py imports from here, so anything
a gradient formula needs that is more than a one-line numpy expression lives in
this file. Everything operates on plain np.ndarray -- nothing here knows about
Tensors or the autograd graph.
"""

import math

import numpy as np

__all__ = [
    "unbroadcast",
    "unreduce",
    "reduced_count",
    "scatter_add",
    "permute_grad",
    "gelu_fn",
    "gelu_grad",
    "log_softmax_stable",
    "layernorm_xhat",
    "layernorm_apply",
    "layernorm_grad_input",
    "cross_entropy_loss",
    "cross_entropy_grad_logits",
]


def unbroadcast(grad, shape):
    """Sum `grad` down to `shape`, undoing numpy broadcasting.

    The codegen wraps every gradient formula in this, so formulas can be
    written naively (`grad * other`) without worrying about a (B, T, C) grad
    flowing back into a (C,) bias.
    """
    grad = np.asarray(grad)
    shape = tuple(shape)
    if grad.shape == shape:
        return grad
    extra = grad.ndim - len(shape)
    if extra > 0:
        grad = grad.sum(axis=tuple(range(extra)))
    axes = tuple(i for i, s in enumerate(shape) if s == 1 and grad.shape[i] != 1)
    if axes:
        grad = grad.sum(axis=axes, keepdims=True)
    return grad.reshape(shape)


def _normalize_dims(dim, ndim):
    if dim is None:
        return tuple(range(ndim))
    dims = (dim,) if isinstance(dim, (int, np.integer)) else tuple(dim)
    return tuple(sorted(d % ndim for d in dims))


def unreduce(grad, shape, dim, keepdim):
    """Undo a reduction: put the collapsed axes back, then broadcast."""
    grad = np.asarray(grad)
    shape = tuple(shape)
    if not keepdim:
        for d in _normalize_dims(dim, len(shape)):
            grad = np.expand_dims(grad, d)
    return np.broadcast_to(grad, shape)


def reduced_count(shape, dim):
    """How many elements each output of a reduction averaged over."""
    shape = tuple(shape)
    n = 1
    for d in _normalize_dims(dim, len(shape)):
        n *= shape[d]
    return n


def scatter_add(shape, index, grad, dtype=None):
    """Gradient of `weight[index]` -- accumulate, since ids repeat."""
    out = np.zeros(shape, dtype=dtype if dtype is not None else grad.dtype)
    np.add.at(out, index, grad)
    return out


def permute_grad(grad, dims):
    dims = tuple(d % np.ndim(grad) for d in dims)
    return np.transpose(grad, np.argsort(dims))


_GELU_C = math.sqrt(2.0 / math.pi)
_GELU_A = 0.044715


def gelu_fn(x):
    return 0.5 * x * (1.0 + np.tanh(_GELU_C * (x + _GELU_A * x**3)))


def gelu_grad(grad, x):
    inner = _GELU_C * (x + _GELU_A * x**3)
    t = np.tanh(inner)
    dinner = _GELU_C * (1.0 + 3.0 * _GELU_A * x * x)
    return grad * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * dinner)


def log_softmax_stable(x, dim=-1):
    shifted = x - np.max(x, axis=dim, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=dim, keepdims=True))


def layernorm_xhat(x, eps):
    mu = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


def layernorm_apply(x, weight, bias, eps):
    return layernorm_xhat(x, eps) * weight + bias


def layernorm_grad_input(grad, x, weight, eps):
    xhat = layernorm_xhat(x, eps)
    inv = 1.0 / np.sqrt(np.var(x, axis=-1, keepdims=True) + eps)
    gh = grad * weight
    return inv * (
        gh
        - np.mean(gh, axis=-1, keepdims=True)
        - xhat * np.mean(gh * xhat, axis=-1, keepdims=True)
    )


def cross_entropy_loss(logits, targets):
    ls = log_softmax_stable(logits, -1)
    flat = ls.reshape(-1, ls.shape[-1])
    t = np.asarray(targets).reshape(-1)
    return np.asarray(-flat[np.arange(t.size), t].mean())


def cross_entropy_grad_logits(grad, logits, targets):
    p = np.exp(log_softmax_stable(logits, -1))
    flat = p.reshape(-1, p.shape[-1]).copy()
    t = np.asarray(targets).reshape(-1)
    flat[np.arange(t.size), t] -= 1.0
    return grad * flat.reshape(p.shape) / t.size
