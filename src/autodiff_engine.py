import numpy as np

from _derivatives_generated import FORWARD, VJP


# ---------------------------------------------------------------------------
# Operation schema
# ---------------------------------------------------------------------------
# {operation: {"params": positional parameter names,
#              "defaults": values filled in when an argument is omitted,
#              "differentiable": indices of params that can receive gradients}}

SCHEMA = {
    "add": {
        "params": ("a", "b"),
        "defaults": {},
        "differentiable": (0, 1),
    },
    "matmul": {
        "params": ("a", "b"),
        "defaults": {},
        "differentiable": (0, 1),
    },
    "relu": {
        "params": ("x",),
        "defaults": {},
        "differentiable": (0,),
    },
    "softmax": {
        "params": ("x", "axis"),
        "defaults": {"axis": -1},
        "differentiable": (0,),
    },
    "sum": {
        "params": ("x", "axis", "keepdims"),
        "defaults": {"axis": None, "keepdims": False},
        "differentiable": (0,),
    },
    "layernorm": {
        "params": ("x", "gamma", "beta", "axis", "eps"),
        "defaults": {"axis": -1, "eps": 1e-5},
        "differentiable": (0, 1, 2),
    },
    "embedding": {
        "params": ("weight", "ids"),
        "defaults": {},
        "differentiable": (0,),
    },
}


# ---------------------------------------------------------------------------
# Forward kernels
#
# Every kernel returns (output, cache); the cache holds whatever the matching
# VJP needs to compute gradients.
# ---------------------------------------------------------------------------

# --- elementwise arithmetic ---

def add_forward(a, b):
    return a + b, (a.shape, b.shape)


def sub_forward(a, b):
    return a - b, (a.shape, b.shape)


def mul_forward(a, b):
    return a * b, (a, b)


def div_forward(a, b):
    return a / b, (a, b)


def neg_forward(x):
    return -x, ()


def pow_forward(x, p):
    return x ** p, (x, p)


# --- shape / linear algebra ---

def matmul_forward(a, b):
    return a @ b, (a, b)


def transpose_forward(x, axes):
    return np.transpose(x, axes), (axes,)


def reshape_forward(x, shape):
    return x.reshape(shape), (x.shape,)


# --- reductions ---

def sum_forward(x, axis, keepdims):
    return np.sum(x, axis=axis, keepdims=keepdims), (x.shape, axis, keepdims)


def mean_forward(x, axis, keepdims):
    out = np.mean(x, axis=axis, keepdims=keepdims)
    count = x.size // out.size
    return out, (x.shape, axis, keepdims, count)


def max_forward(x, axis, keepdims):
    out = np.max(x, axis=axis, keepdims=True)
    mask = (x == out)
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out, (mask, axis, keepdims)


# --- unary math ---

def exp_forward(x):
    out = np.exp(x)
    return out, (out,)


def log_forward(x):
    return np.log(x), (x,)


def sqrt_forward(x):
    out = np.sqrt(x)
    return out, (out,)


def tanh_forward(x):
    out = np.tanh(x)
    return out, (out,)


# --- activations ---

_GELU_C = np.sqrt(2.0 / np.pi)


def relu_forward(x):
    mask = x > 0
    return np.where(mask, x, 0.0), (mask,)


def gelu_forward(x):
    inner = _GELU_C * (x + 0.044715 * x ** 3)
    return 0.5 * x * (1.0 + np.tanh(inner)), (x,)


def softmax_forward(x, axis):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    probs = e / np.sum(e, axis=axis, keepdims=True)
    return probs, (probs, axis)


def log_softmax_forward(x, axis):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    out = shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    return out, (out, axis)


# --- layers ---

def layernorm_forward(x, gamma, beta, axis, eps):
    mu = np.mean(x, axis=axis, keepdims=True)
    xc = x - mu
    var = np.mean(xc ** 2, axis=axis, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xhat = xc * inv
    return xhat * gamma + beta, (xhat, inv, gamma, axis)


def dropout_forward(x, p, training, rng):
    if not training or p == 0.0:
        return x, (None,)
    keep = rng.random(x.shape) >= p
    scale = 1.0 / (1.0 - p)
    return x * keep * scale, (keep, scale)


# --- indexing / assembly ---

def getitem_forward(x, idx):
    return x[idx], (x.shape, idx)


def embedding_forward(weight, ids):
    return weight[ids], (weight.shape, ids)


def concat_forward(xs, axis):
    sizes = tuple(x.shape[axis] for x in xs)
    return np.concatenate(xs, axis=axis), (sizes, axis)


def masked_fill_forward(x, mask, value):
    return np.where(mask, value, x), (mask,)


# --- losses ---

def cross_entropy_forward(logits, targets, axis):
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    logZ = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    log_probs = shifted - logZ
    n = targets.size
    picked = np.take_along_axis(log_probs, np.expand_dims(targets, axis), axis)
    loss = -np.mean(picked)
    return loss, (np.exp(log_probs), targets, axis, n)


FORWARD = {
    "add":           add_forward,
    "sub":           sub_forward,
    "mul":           mul_forward,
    "div":           div_forward,
    "neg":           neg_forward,
    "pow":           pow_forward,

    "matmul":        matmul_forward,
    "transpose":     transpose_forward,
    "reshape":       reshape_forward,

    "sum":           sum_forward,
    "mean":          mean_forward,
    "max":           max_forward,

    "exp":           exp_forward,
    "log":           log_forward,
    "sqrt":          sqrt_forward,
    "tanh":          tanh_forward,

    "relu":          relu_forward,
    "gelu":          gelu_forward,
    "softmax":       softmax_forward,
    "log_softmax":   log_softmax_forward,

    "layernorm":     layernorm_forward,

    "getitem":       getitem_forward,
    "embedding":     embedding_forward,
    "concat":        concat_forward,

    "masked_fill":   masked_fill_forward,
    "dropout":       dropout_forward,
    "cross_entropy": cross_entropy_forward,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bind(name, args, kwargs):
    """Resolve positional/keyword arguments into the schema's parameter order."""
    spec = SCHEMA[name]
    params = spec["params"]
    defaults = spec.get("defaults", {})

    if len(args) > len(params):
        raise TypeError(f"{name}: expected at most {len(params)} args, but got {len(args)}")

    bound = list(args)
    for p in params[len(args):]:
        if p in kwargs:
            bound.append(kwargs.pop(p))
        elif p in defaults:
            bound.append(defaults[p])
        else:
            raise TypeError(f"{name}: missing required argument '{p}'")

    if kwargs:
        raise TypeError(f"{name}: unexpected keyword(s) {sorted(kwargs)}")

    return tuple(bound)


def topo_order(root):
    """Iterative post-order traversal of the graph reachable from `root`."""
    topo = []
    visited = set()
    stack = [(root, False)]

    while stack:
        tensor, expanded = stack.pop()

        if expanded:
            topo.append(tensor)
            continue
        if id(tensor) in visited:
            continue
        visited.add(id(tensor))

        stack.append((tensor, True))
        if tensor._node is not None:
            for parent in tensor._node.parents:
                if isinstance(parent, Tensor) and id(parent) not in visited:
                    stack.append((parent, False))

    return topo


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

class Node:
    """One operation in the graph: which op produced a tensor, from what parents."""

    def __init__(self, operation: str, parents: list, cache, needs: list[bool]):
        self.operation = operation
        self.parents = parents
        self.cache = cache
        self.needs = needs


class Tensor:
    """
    data: numpy array
    requires_grad: boolean
    grad: gradient object for this tensor
    _node: node that contains operations (add, sub, multi, etc) of parents
    """

    def __init__(self, data, requires_grad, grad=None, _node=None):
        self.data = data
        self.grad = grad  # numpy array
        self.requires_grad = requires_grad
        self._node = _node

    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.data)
        self.grad = grad

        topo = topo_order(self)

        for tensor in topo:
            if tensor.requires_grad and tensor.grad is None:
                tensor.grad = np.zeros_like(tensor.data)

        for tensor in reversed(topo):
            if tensor._node is None:
                continue

            upstream_grad = tensor.grad
            vjp = VJP[tensor._node.operation](upstream_grad, tensor._node.cache)

            for parent, needs, current_tensor_grad in zip(
                tensor._node.parents, tensor._node.needs, vjp
            ):
                if needs:
                    parent.grad += current_tensor_grad


    def zero_grad(self):
        self.grad = None
        return

    @staticmethod
    def apply(name, *args, **kwargs):
        bound = _bind(name, args, kwargs)  # returns tensors

        raw = []
        for tensor in bound:
            if isinstance(tensor, Tensor):
                raw.append(tensor.data)
            else:
                raw.append(tensor)
        raw = tuple(raw)

        out, cache = FORWARD[name](*raw)

        parents, needs_grad_tensors = [], []
        for i in SCHEMA[name]["differentiable"]:
            tensor = bound[i]
            parents.append(tensor)
            needs_grad_tensors.append(isinstance(tensor, Tensor) and tensor.requires_grad)

        if any(needs_grad_tensors):
            node = Node(
                operation=name,
                parents=parents,
                cache=cache,
                needs=needs_grad_tensors,
            )
            return Tensor(data=out, requires_grad=True, _node=node)

        return Tensor(data=out, requires_grad=False)  # layers that dont need grad

    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def __repr__(self):
        return "Tensor(shape=(" + str(self.shape) + ",requires_grad=" + str(self.requires_grad) + ")"
