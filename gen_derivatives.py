# gen_derivatives.py
import yaml

HEADER = '''"""
genrerated by gen_derivatives.py. look at derivatives.yaml to change stuff
"""
import numpy as np
from derivative_helpers import *
'''

def load_spec(path="derivatives.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def gen_forward(entry):
    name, args = entry["name"], entry["args"]
    defaults = entry.get("defaults", {})
    sig = ", ".join(f"{a}={defaults[a]!r}" if a in defaults else a for a in args)
    save = ", ".join(f"'{a}': {a}" for a in args)
    return (
        f"def {name}_forward({sig}):\n"
        f"    result = {entry['forward']}\n"
        f"    cache = {{{save}, 'result': result}}\n"
        f"    return result, cache"
    )

def gen_backward(entry):
    name, args = entry["name"], entry["args"]
    nondiff = set(entry.get("nondiff", []))
    diff_args = [a for a in args if a not in nondiff]
    deriv = entry["derivatives"]

    unpack = ", ".join(args + ["result"])
    fetch = ", ".join(f"cache['{a}']" for a in args + ["result"])
    lines = [f"def {name}_backward(grad, cache):", f"    {unpack} = {fetch}"]
    for a in diff_args:
        lines.append(f"    d_{a} = unbroadcast({deriv[a]}, np.shape({a}))")
    ret = ", ".join(f"d_{a}" for a in diff_args)
    lines.append(f"    return ({ret},)" if len(diff_args) == 1 else f"    return ({ret})")
    return "\n".join(lines)

def generate(spec_path="derivatives.yaml", out_path="_derivatives_generated.py"):
    entries = load_spec(spec_path)
    chunks = [HEADER]
    names = [e["name"] for e in entries]
    for entry in entries:
        chunks.append(gen_forward(entry))
        chunks.append(gen_backward(entry))
    chunks.append("FORWARD = {" + ", ".join(f"'{n}': {n}_forward" for n in names) + "}")
    chunks.append("VJP = {" + ", ".join(f"'{n}': {n}_backward" for n in names) + "}")
    with open(out_path, "w") as f:
        f.write("\n\n".join(chunks))

if __name__ == "__main__":
    generate()