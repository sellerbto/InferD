import io
import torch
import grpc
import asyncio
import uuid
from typing import Dict, List, Tuple, Optional

from models.qwen3.proto import qwen3_pb2, qwen3_pb2_grpc

class ChainRPCQwen3Client:
    """
    Enhanced RPC client that implements chain-based inference with server-side KV cache.
    """

    def __init__(self, bootstrap_nodes: List[Tuple[str, int]], num_stages: int = 4):
        self.bootstrap_nodes = bootstrap_nodes
        self.num_stages = num_stages
        # gRPC options
        self.options = [
            ("grpc.max_send_message_length", 200 * 1024 * 1024),
            ("grpc.max_receive_message_length", 200 * 1024 * 1024),
        ]
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            ("mps" if torch.backends.mps.is_available() else "cpu")
        )

    def serialize_tensor(self, tensor: torch.Tensor) -> bytes:
        buf = io.BytesIO()
        torch.save(tensor.detach().cpu(), buf)
        return buf.getvalue()

    def deserialize_tensor(self, data: bytes) -> torch.Tensor:
        buf = io.BytesIO(data)
        return torch.load(buf, map_location=self.device)

    def create_stub(self, host: str, port: int) -> qwen3_pb2_grpc.Qwen3LayerStub:
        channel = grpc.insecure_channel(f"{host}:{port}", options=self.options)
        return qwen3_pb2_grpc.Qwen3LayerStub(channel)

    async def find_optimal_chain(self, client_id: str = "chain_client") -> List[Tuple[str, int]]:
        # Attempt to get optimal chain from bootstrap nodes
        for ip, port in self.bootstrap_nodes:
            try:
                stub = self.create_stub(ip, port)
                req = qwen3_pb2.GetOptimalChainRequest(
                    num_stages=self.num_stages, client_id=client_id
                )
                resp = stub.GetOptimalChain(req, timeout=30.0)
                if resp.success:
                    return [(node.ip, node.port) for node in resp.chain]
            except Exception:
                continue
        # fallback round-robin
        return [
            (self.bootstrap_nodes[i % len(self.bootstrap_nodes)][0],
             self.bootstrap_nodes[i % len(self.bootstrap_nodes)][1])
            for i in range(self.num_stages)
        ]

    async def forward_through_chain(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        cache_position: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        session_id: str = "",
        timeout: float = 60.0,
        start_stage: int = 0,
        use_cache: bool = True,
    ) -> torch.Tensor:
        if not session_id:
            session_id = str(uuid.uuid4())

        # discover chain asynchronously
        optimal_chain = await self.find_optimal_chain(session_id)
        print(f'Optimal chain: {optimal_chain}')

        for stage, (ip, port) in enumerate(optimal_chain[start_stage:], start_stage):
            stub = self.create_stub(ip, port)
            req = qwen3_pb2.LayerRequest(
                hidden_states=qwen3_pb2.TensorBlob(data=self.serialize_tensor(hidden_states)),
                attention_mask=None if attention_mask is None else qwen3_pb2.TensorBlob(data=self.serialize_tensor(attention_mask)),
                cache_position=qwen3_pb2.TensorBlob(data=self.serialize_tensor(cache_position)),
                cos_embedding=qwen3_pb2.TensorBlob(data=self.serialize_tensor(cos)),
                sin_embedding=qwen3_pb2.TensorBlob(data=self.serialize_tensor(sin)),
                session_id=session_id,
                stage=stage,
            )
            resp = stub.ProcessLayer(req, timeout=timeout)
            hidden_states = self.deserialize_tensor(resp.hidden_states.data)

        return hidden_states

    def forward_through_chain_sync(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        cache_position: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        session_id: str = "",
        timeout: float = 60.0,
        start_stage: int = 0,
        use_cache: bool = True,
    ) -> torch.Tensor:
        return asyncio.run(
            self.forward_through_chain(
                hidden_states,
                attention_mask,
                cache_position,
                cos,
                sin,
                session_id=session_id,
                timeout=timeout,
                start_stage=start_stage,
                use_cache=use_cache,
            )
        )

    def forward_incremental(
        self,
        hidden_states: torch.Tensor,
        cache_position: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        session_id: str,
        timeout: float = 60.0,
    ) -> torch.Tensor:
        """
        Send only new embeddings; rely on server-side KV cache for past states.
        """
        return self.forward_through_chain_sync(
            hidden_states=hidden_states,
            attention_mask=None,
            cache_position=cache_position,
            cos=cos,
            sin=sin,
            session_id=session_id,
            timeout=timeout,
            start_stage=0,
            use_cache=True,
        )
