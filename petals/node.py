import asyncio
from kademlia_client import DistributedHashTableServer
from aiohttp import web
import aiohttp
import torch
import grpc
import io

from task import QwenTask
from node_info import NodeInfo
from path_finder import PathFinder
from balance import Balancer
from task_scheduler import TaskScheduler
from models.qwen3.server.qwen3_server_module import Qwen3Server

from models.qwen3.proto import qwen3_pb2
from models.qwen3.proto import qwen3_pb2_grpc
from utils import deserialize_tensor, get_start_end_layer_by_stage, create_stub


class Node:
    def __init__(self,
                 node_ip,
                 node_port,
                 name,
                 model_name,
                 initial_stage,
                 num_stages: int,
                 capacity: int,
                 dht: DistributedHashTableServer,
                 rebalance_period: int = 2):

        print(f'Type initial stage: {type(initial_stage)}')
        initial_stage = int(initial_stage)

        self.node_info = NodeInfo(
            id=f'{node_ip}:{node_port}',
            ip=node_ip,
            port=node_port,
            name=name,
            model_name=model_name,
            num_stages=num_stages,
            capacity=capacity,
            rebalance_period=rebalance_period,
            stage=initial_stage
        )

        self.dht = dht
        self.task_scheduler = TaskScheduler(self.node_info, self.dht)
        self.balancer = Balancer(self.dht, self.node_info, self.task_scheduler, 5)
        self.path_finder = PathFinder(self.dht, self.node_info, self.balancer)
        self._rebalance_task = None

        self.server = self.init_server(initial_stage)
        self.options = [
            ("grpc.max_send_message_length", 200 * 1024 * 1024),
            ("grpc.max_receive_message_length", 200 * 1024 * 1024),
        ]


    def deserialize_tensor(self, blob: qwen3_pb2.TensorBlob):
        buf = io.BytesIO(blob.data)
        return torch.load(buf, map_location=self.server.device)

    def serialize_tensor(self, tensor: torch.Tensor):
        buf = io.BytesIO()
        torch.save(tensor.detach().cpu(), buf)
        return buf.getvalue()

    def init_server(self, stage):
        start_layer, end_layer = get_start_end_layer_by_stage(stage)
        return Qwen3Server(start_layer, end_layer)

    async def ProcessLayer(self, request, context):
        print(f'REQUEST RECEIVED')
        stage = request.stage
        session_id = request.session_id

        # Check if this node can handle the request stage
        if stage != self.node_info.stage:
            print(f'Error: Node at stage {self.node_info.stage} cannot handle stage {stage}')
            return await self.send_to_next_node_direct(request, stage)

        hidden = self.deserialize_tensor(request.hidden_states).to(self.server.device)
        attn_mask = self.deserialize_tensor(request.attention_mask).to(
            self.server.device
        )
        cache_pos = self.deserialize_tensor(request.cache_position).to(
            self.server.device
        )
        cos = self.deserialize_tensor(request.cos_embedding).to(self.server.device)
        sin = self.deserialize_tensor(request.sin_embedding).to(self.server.device)

        out = self.server.forward(
            session_id=session_id,
            hidden_states=hidden,
            attention_mask=attn_mask,
            cache_position=cache_pos,
            position_embeddings=(cos, sin),
        )

        return qwen3_pb2.LayerResponse(
            hidden_states=qwen3_pb2.TensorBlob(data=self.serialize_tensor(out))
        )


    async def GetOptimalChain(self, request, context):
        try:
            num_stages = request.num_stages
            client_id = request.client_id

            print(f"Getting optimal chain for client {client_id}, num_stages={num_stages}")

            # Use the private _find_best_chain method from PathFinder
            optimal_chain = await self.path_finder._find_best_chain()

            # Convert to protobuf NodeInfo format
            chain_nodes = []
            for i, (ip, port) in enumerate(optimal_chain):
                chain_nodes.append(qwen3_pb2.NodeInfo(
                    ip=ip,
                    port=port,
                    stage=i
                ))

            print(f"Optimal chain computed: {[(node.ip, node.port, node.stage) for node in chain_nodes]}")

            return qwen3_pb2.GetOptimalChainResponse(
                chain=chain_nodes,
                success=True,
                error_message=""
            )

        except Exception as e:
            error_msg = f"Failed to compute optimal chain: {str(e)}"
            print(f"Error in GetOptimalChain: {error_msg}")

            return qwen3_pb2.GetOptimalChainResponse(
                chain=[],
                success=False,
                error_message=error_msg
            )

    async def rebalance_task(self):
        while True:
            await self.rebalance()
            await asyncio.sleep(self.node_info.rebalance_period)

    async def change_stage(self, new_stage):
        record = await self.dht.get(str(self.node_info.stage)) or {}
        record.pop(self.node_info.id, None)

        self.server = self.init_server(new_stage)
        await self.dht.set(str(self.node_info.stage), record)

        self.node_info.set_stage(new_stage)
        await self.task_scheduler.announce()

    async def rebalance(self):
        print('Rebalancing...')
        return await self.balancer.rebalance()

    async def handle_reassign(self, request):
        data = await request.json()
        new_stage = data.get('to')
        await self.change_stage(new_stage)
        print(f'Node {self.node_info.id} changed stage from {self.node_info.stage} to {new_stage}, {type(self.node_info.stage)}, {type(new_stage)}')
        return web.json_response({
            'node_info.id': self.node_info.id,
            'new_stage': self.node_info.stage,
            'status': 'reassigned'
        })

    async def post(self, url, payload):
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload) as resp:
                    return await resp.json()
            except Exception as e:
                print(f"Error sending to {url}: {e}")
                return None

    async def run_task(self, task):
        print(f'Node {self.node_info.id} running task {task.id}')
        await self.task_scheduler.run_task(task)
        cur_stage = task.stage
        task_result = task.get_result()

        print(f'Node {self.node_info.id}, cur_stage = {cur_stage}, num_stages = {self.node_info.num_stages}')

        return qwen3_pb2.LayerResponse(
            hidden_states=qwen3_pb2.TensorBlob(data=self.serialize_tensor(task_result))
        )

    async def start(self, initial_stage: int, rebalance=True):
        self.node_info.set_stage(initial_stage)
        if rebalance:
            # self.rebalance_task()
            self._rebalance_task = asyncio.create_task(self.rebalance_task())
            print("⚙️ grebalance_task created:", self._rebalance_task, flush=True)
        else:
            self._rebalance_task = None

    async def stop(self):
        if self._rebalance_task:
            self._rebalance_task.cancel()

class NodeServicer(qwen3_pb2_grpc.Qwen3LayerServicer):
    def __init__(self, node: Node):
        self.node = node

    async def ProcessLayer(self, request, context):
        # Delegate to the Node's ProcessLayer implementation
        return await self.node.ProcessLayer(request, context)

    async def ProcessChain(self, request, context):
        # Delegate to the Node's ProcessChain implementation
        return await self.node.ProcessChain(request, context)

    async def GetOptimalChain(self, request, context):
        # Delegate to the Node's GetOptimalChain implementation
        return await self.node.GetOptimalChain(request, context)
