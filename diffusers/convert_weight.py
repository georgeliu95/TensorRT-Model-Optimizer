from argparse import ArgumentParser
from collections import defaultdict
import numpy as np
import onnx
from onnx import numpy_helper
import os
import sys
from typing import Dict, List


def get_tensor_producer_nodes(
    graph: onnx.onnx_ml_pb2.GraphProto,
) -> Dict[str, onnx.onnx_ml_pb2.NodeProto]:
    """Returns a dictionary of tensor name and their producer node object mapping.

    Note. we create a special Root type node as external inputs producer for ease of implementation.
    """
    # Create a dictionary to store tensor producer nodes
    tensor_producers = defaultdict(None)

    # Special Root type producer node
    root_node = onnx.helper.make_node(
        op_type="Root",
        inputs=[],
        outputs=[i.name for i in graph.input],
        name="root_0",
    )

    input_names = [graph_input.name for graph_input in graph.input]
    initializer_names = [initializer.name for initializer in graph.initializer]
    external_input_names = list(np.setdiff1d(input_names, initializer_names))

    # Note. We are marking external inputs as non-constant by adding a parent,
    # so that we can quantize the first node of the graph if appropriate
    for graph_input in external_input_names:
        tensor_producers[graph_input] = root_node

    # Traverse the graph to find producer nodes for each tensor
    for node in graph.node:
        for output_name in node.output:
            tensor_producers[output_name] = node

    return tensor_producers


def get_tensor_consumer_nodes(
    graph: onnx.onnx_ml_pb2.GraphProto,
) -> Dict[str, List[onnx.onnx_ml_pb2.NodeProto]]:
    """Returns a dictionary of tensor name and their consumer node object mapping."""
    # Create a dictionary to store tensor consumer nodes
    tensor_consumers = defaultdict(list)

    # Traverse the graph to find consumer nodes for each tensor
    for node in graph.node:
        for input_name in node.input:
            tensor_consumers[input_name].append(node)

    return tensor_consumers


def convert_to_w8(onnx_path: str, output_onnx_path: str):
    onnx_model = onnx.load(onnx_path)
    graph = onnx_model.graph
    initializers = graph.initializer
    tensor_producers = get_tensor_producer_nodes(graph)
    tensor_consumers = get_tensor_consumer_nodes(graph)

    def _get_initializer_index(name: str):
        for idx, init in enumerate(initializers):
            if init.name == name:
                return idx

    def _convert(node: onnx.onnx_ml_pb2.NodeProto):
        print(f"Processing {node.name}")
        idx1 = _get_initializer_index(node.input[0])
        w = initializers[idx1]
        dtype = onnx.helper.tensor_dtype_to_np_dtype(w.data_type)
        w32 = np.frombuffer(w.raw_data, dtype=dtype).reshape(w.dims).astype(np.float32)

        idx2 = _get_initializer_index(node.input[1])
        if idx2 is not None:
            y_scale = initializers[idx2]    # TODO: Dangling scale and zero-point node after conversion
            dtype = onnx.helper.tensor_dtype_to_np_dtype(y_scale.data_type)
            np_y_scale = np.array(y_scale.float_data, dtype=dtype).reshape(y_scale.dims).astype(np.float32)
        else:
            producer_node = tensor_producers[node.input[1]]
            y_scale = producer_node.attribute[0].t
            dtype = onnx.helper.tensor_dtype_to_np_dtype(y_scale.data_type)
            np_y_scale = np.frombuffer(y_scale.raw_data, dtype=dtype).reshape(y_scale.dims).astype(np.float32)

        do_transpose = w32.shape[-1] != np_y_scale.shape[0]
        if do_transpose:
            # Gemm has co as first dimension
            w32 = np.transpose(w32)

        w32_scaled_clipped = np.asarray((w32 / np_y_scale).round())
        np.clip(w32_scaled_clipped, -128, 127, out=w32_scaled_clipped)
        if do_transpose:
            w32_scaled_clipped = np.transpose(w32_scaled_clipped)

        w8 = numpy_helper.from_array(w32_scaled_clipped.astype("int8"), w.name)
        initializers[idx1].CopyFrom(w8)

        return idx2, _get_initializer_index(node.input[2])


    dangling_q_indices = []
    dangling_init_indices = []

    for node_idx, node in enumerate(graph.node):
        if node.op_type == "QuantizeLinear":
            weight_name = node.input[0]

            # Const input to quantize linear means weighted layer
            if node.input[0] not in tensor_producers:
                scale_init_idx, zp_init_idx = _convert(node)
                dangling_q_indices.append(node_idx)
                dangling_init_indices.extend([scale_init_idx, zp_init_idx])

                # Update following DQ nodes input name
                dq_node = tensor_consumers[node.output[0]][0]
                assert dq_node.op_type == "DequantizeLinear"
                dq_node.input[0] = weight_name

    # Remove Q nodes
    for node_idx in sorted(dangling_q_indices, reverse=True):
        del graph.node[node_idx]

    # Save the modified model
    print(f"W8 model is saved as {output_onnx_path}")
    onnx.save(onnx_model, output_onnx_path)


argparser = ArgumentParser("Convert the weight of a DequantizedLinear node to int8.")
argparser.add_argument(
    "--onnx",
    type=str,
    help="Onnx model path to convert the weight.",
)
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_weight.py --onnx=<MODEL_NAME>.onnx")
        sys.exit(1)

    args = argparser.parse_args()
    onnx_path = args.onnx
    model_name = os.path.splitext(os.path.basename(onnx_path))[0]
    output_onnx_path = os.path.join(os.path.dirname(onnx_path), model_name + ".w8.onnx")

    convert_to_w8(onnx_path, output_onnx_path)
