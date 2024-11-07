import onnx
import numpy as np
from onnx import numpy_helper

# Load the model
model = onnx.load("forklift.onnx")

# Convert any INT64 weights to INT32
for initializer in model.graph.initializer:
    if initializer.data_type == onnx.TensorProto.INT64:
        tensor = numpy_helper.to_array(initializer).astype(np.int32)
        initializer.CopyFrom(numpy_helper.from_array(tensor, initializer.name))
for node in model.graph.node:
    if node.op_type == "Reduce" and not any(attr.name == "axes" for attr in node.attribute):
        axes_attr = onnx.helper.make_attribute("axes", [1])  # Example axis
        node.attribute.append(axes_attr)

# Save the updated model
onnx.save(model, "new_model.onnx")
