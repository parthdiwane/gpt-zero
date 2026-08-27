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
        "params": ("x", "gamma", "beta", "eps"),
        "defaults": {"eps": 1e-5},
        "differentiable": (0, 1, 2),
    },
    "embedding": {
        "params": ("weight", "ids"),
        "defaults": {},
        "differentiable": (0,),
    },
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

    def __add__(self, rhs: Tensor):
        return self.data + rhs.data
    def __mul__(self, rhs: Tensor):
        return np.dot(self.data, rhs.data)
    def __matmul__(self, rhs: Tensor):
        return np.matmul(self.data, rhs.data)
    def __neg__(self):
        return -self.data
    def __pow__(self, rhs: int):
        return self.data ** rhs
    def __truediv__(self, rhs: Tensor):
        return self.data / rhs
    def __setitem__(self, row, col, value):
        self.data[row][col] = value
    def __getitem__(self, row, col):
        return self.data[row][col]
    def __eq__(self, rhs: Tensor):
        return self.data == rhs.data

    
        