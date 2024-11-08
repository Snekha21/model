import onnx
import numpy as np
from onnx import helper
# Load the model
model = onnx.load("new_model.onnx")


import onnx
import json


# Create a list of dictionaries to store node attributes
node_attributes = []

# Helper function to convert ONNX attribute values to a serializable format
def convert_attribute_value(attr):
    if attr.type == onnx.AttributeProto.INTS:
        return list(attr.ints)  # Convert RepeatedScalarContainer to list
    elif attr.type == onnx.AttributeProto.FLOATS:
        return list(attr.floats)
    elif attr.type == onnx.AttributeProto.STRING:
        return attr.s.decode('utf-8')  # Decode bytes to string
    elif attr.type == onnx.AttributeProto.TENSOR:
        return str(attr.t)  # You can modify this part to convert tensor objects properly
    else:
        return None  # Handle other types as needed

# Iterate through each node and extract attributes
for node in model.graph.node:
    node_info = {
        'node_name': node.name,
        'op_type': node.op_type,
        'attributes': {}
    }
    # Extract the attributes of the node
    for attr in node.attribute:
        node_info['attributes'][attr.name] = {
            'type': attr.type,
            'value': convert_attribute_value(attr)
        }
    node_attributes.append(node_info)

# Save the extracted node attributes as a JSON file
with open("node_attributes.json", "w") as json_file:
    json.dump(node_attributes, json_file, indent=4)

print("Node attributes saved to node_attributes.json")


