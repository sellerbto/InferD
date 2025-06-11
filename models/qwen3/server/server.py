import os
import io
import torch
import grpc
from concurrent import futures
from proto import qwen3_pb2
from proto import qwen3_pb2_grpc
from .qwen3_server_module import Qwen3Server
import argparse


class Qwen3LayerServicer(qwen3_pb2_grpc.Qwen3LayerServicer):
    def __init__(self, start_layer: int, end_layer: int):
        self.server_module = Qwen3Server(start_layer, end_layer)

    def deserialize_tensor(self, blob: qwen3_pb2.TensorBlob):
        buf = io.BytesIO(blob.data)
        return torch.load(buf, map_location=self.server_module.device)

    def serialize_tensor(self, tensor: torch.Tensor):
        buf = io.BytesIO()
        torch.save(tensor.detach().cpu(), buf)
        return buf.getvalue()

    def ProcessLayer(self, request, context):
        try:
            hidden = self.deserialize_tensor(request.hidden_states).to(self.server_module.device)
            attn_mask = self.deserialize_tensor(request.attention_mask).to(
                self.server_module.device
            )
            cache_pos = self.deserialize_tensor(request.cache_position).to(
                self.server_module.device
            )
            cos = self.deserialize_tensor(request.cos_embedding).to(self.server_module.device)
            sin = self.deserialize_tensor(request.sin_embedding).to(self.server_module.device)
        except Exception as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"Failed to deserialize tensors: {e}")

        sid = request.session_id or None

        try:
            out = self.server_module.send(
                session_id=sid,
                hidden_states=hidden,
                attention_mask=attn_mask,
                cache_position=cache_pos,
                position_embeddings=(cos, sin),
            )
        except Exception as e:
            context.abort(grpc.StatusCode.INTERNAL, f"Error in model forward: {e}")

        return qwen3_pb2.LayerResponse(
            hidden_states=qwen3_pb2.TensorBlob(data=self.serialize_tensor(out))
        )


def serve():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_layer", type=int, required=True)
    parser.add_argument("--end_layer", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    options = [
        ("grpc.max_receive_message_length", 100 * 1024 * 1024),
        ("grpc.max_send_message_length", 100 * 1024 * 1024),
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=options)
    qwen3_pb2_grpc.add_Qwen3LayerServicer_to_server(
        Qwen3LayerServicer(args.start_layer, args.end_layer), server
    )
    server.add_insecure_port(f"[::]:{args.port}")
    server.start()
    print(f"gRPC server listening on port {args.port}, layers {args.start_layer}-{args.end_layer}")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
