import onnx
import numpy as np
from onnx import helper
# Load the model
model = onnx.load("new_model.onnx")


# Load model

# Find the relevant operation and manually add axes
for node in model.graph.node:
    if node.op_type == "Transpose":
        if "perm" not in node.attribute:
            # Add a permutation if missing
            node.attribute.append(helper.make_attribute("perm", [0, 2, 3, 1]))

# Save updated model
onnx.save(model, "new_model.onnx")

