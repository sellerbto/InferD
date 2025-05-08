import asyncio
from kademlia_client import DistributedHashTableServer
from aiohttp import ClientSession
from task import *
from utils import parse_ip_port
from node_info import NodeInfo
from balance import Balancer
from utils import *

class PathFinder:
    def __init__(self, 
                dht : DistributedHashTableServer,
                node_info : NodeInfo, 
                balancer: Balancer):
        self.dht = dht
        self.node_info = node_info
        self.balancer = balancer

    async def find_best_chain(self) -> list:
        # todo: D^* algorithm...
        chain = []
        for stage in range(self.node_info.num_stages):
            record = await self.dht.get(str(stage)) or {}
            if not record:
                await self.balancer.rebalance()
                record = await self.dht.get(str(stage)) or {}
                if not record:
                    raise RuntimeError(f"No available servers for stage {stage} after rebalance")
            ip_port, _ = min(record.items(), key=lambda x: x[1]['load'])
            ip, port = parse_ip_port(ip_port)
            chain.append((ip, port))
        return chain

    async def find_best_node(self, stage: int, retry: int = 3) -> int:
        # todo: D^* algorithm...
        if retry == 0:
            print(f'Try to reassign node')
            dht_map = await self.dht.get_all() 
            lmin, lmax, smin, smax = await min_max_load_stage(self.node_info, dht_map)
            print(lmin, lmax, smin, smax)
    
            potential_nodes_to_reassign = dht_map[str(smin)].keys()
            print(f'potential_nodes_to_reassign = {potential_nodes_to_reassign}')
            ip, port, old_stage = None, None, None

            min_load_stages = get_min_load_stages(dht_map, lmin)
            min_load_stages = min_load_stages - set([stage])
            print(f'min_load_stages = {min_load_stages}')
            for st in min_load_stages:
                nodes_info = dht_map[str(st)]
                if nodes_info and int(stage) != int(st):
                    ip, port = parse_ip_port(next(iter(nodes_info.keys())))
                    old_stage = st
                    break

            if None in [ip, port, old_stage]:
                for st, nodes_info in dht_map.items():
                    if nodes_info and int(st) != int(stage):
                        ip, port = parse_ip_port(next(iter(nodes_info.keys())))
                        old_stage = st
                        break
            if None in [ip, port, old_stage]:
                raise Exception('No nodes to reassign')
            print(f'Try to reassign node {ip}:{port} from {old_stage} to {stage}')
            await self.reassign_node(ip, port, stage)
            # raise Exception('3 tries to find node with stage {stage}, but not found...')

        record = await self.dht.get(str(stage)) or {}
        if not record:
            print(f'No peers available for stage {stage}. Try to rebalance this node with stage = {self.node_info.stage} to stage {stage}')
            # надо какую-то ноду назначить на этот стейдж, сейчас у текущей ноды берется rebalance в надежде что эта нода перестроится на пустой стедж
            # raise RuntimeError(f"No peers available for stage {stage}") 
            old_stage = self.node_info.stage
            rebalanced = await self.balancer.rebalance() 
            rebalance_verbose = f'stage changed from {old_stage} to {self.node_info.stage}' if rebalanced else 'nothing changed'
            print(f'Rebalancing result: {rebalance_verbose}')
            print(f'Retry')
            return await self.find_best_node(stage, retry-1)
        else:
            print(f'Found {stage}: {record}')
        ip_port, _ = min(record.items(), key=lambda x: x[1]['load'])
        # await asyncio.sleep(1)
        return parse_ip_port(ip_port)
    
    async def reassign_node(self, ip: str, http_port: int, to_stage: int):
        url = f'http://{ip}:{http_port}/reassign'
        async with ClientSession() as session:
            async with session.post(url, json={'to': to_stage}) as resp:
                return await resp.json() 