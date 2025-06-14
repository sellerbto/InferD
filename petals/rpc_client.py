import io
import torch
import grpc
from models.qwen3.proto import qwen3_pb2, qwen3_pb2_grpc


class RPCQwen3Client:
    def __init__(self, server_addrs):
        """
        server_addrs: list из кортежей (host, port), например [("localhost", 50051), ...]
        """
        options = [
            ("grpc.max_send_message_length", 200 * 1024 * 1024),
            ("grpc.max_receive_message_length", 200 * 1024 * 1024),
        ]
        self.stubs = []
        for host, port in server_addrs:
            channel = grpc.insecure_channel(f"{host}:{port}", options=options)
            stub = qwen3_pb2_grpc.Qwen3LayerStub(channel)
            self.stubs.append(stub)
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

    def serialize_tensor(self, tensor: torch.Tensor) -> bytes:
        buf = io.BytesIO()
        torch.save(tensor.detach().cpu(), buf)
        return buf.getvalue()

    def deserialize_tensor(self, data: bytes) -> torch.Tensor:
        buf = io.BytesIO(data)
        return torch.load(buf, map_location=self.device)

    def forward_through_chain(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        session_id: str = "",
        timeout: float = 30.0,
    ) -> torch.Tensor:

        stubs_chain = find_best_chain()
        for i, stub in enumerate(stubs_chain):
            req = qwen3_pb2.LayerRequest(
                hidden_states=qwen3_pb2.TensorBlob(data=self.serialize_tensor(hidden_states)),
                attention_mask=qwen3_pb2.TensorBlob(data=self.serialize_tensor(attention_mask)),
                cache_position=qwen3_pb2.TensorBlob(data=self.serialize_tensor(cache_position)),
                cos_embedding=qwen3_pb2.TensorBlob(data=self.serialize_tensor(cos)),
                sin_embedding=qwen3_pb2.TensorBlob(data=self.serialize_tensor(sin)),
                session_id=session_id,
                stage=i
            )
            resp = stub.ProcessLayer(req, timeout=timeout)
            hidden_states = self.deserialize_tensor(resp.hidden_states.data)
        return hidden_states
