import asyncio
from kademlia_client import DistributedHashTableServer
from aiohttp import web, ClientSession
from task import *
from config import port_shift
import traceback
from utils import parse_ip_port
from node_info import NodeInfo
from task_scheduler import TaskScheduler
from utils import *

class Balancer:
    def __init__(self, dht: DistributedHashTableServer, node_info: NodeInfo,
    task_scheduler: TaskScheduler,
    rebalance_period: int):
        self.node_info = node_info
        self.dht = dht
        self.task_scheduler = task_scheduler
        self.rebalance_period = rebalance_period

    async def rebalance(self):
        # await asyncio.sleep(self.rebalance_period)
        # return
        try:
            await asyncio.sleep(self.rebalance_period)
            qi = self.measure_load()
            # print(f'LOAD = {qi}')
            node_id = self.node_info.id
            stage = self.node_info.stage
            record = await self.dht.get(str(stage)) or {}
            
            record[node_id] = {'load': qi, 'cap': self.node_info.capacity}
            await self.dht.set(str(stage), record)

            dht_map = await self.dht.get_all()
            lmin, lmax, smin, smax = await min_max_load_stage(self.node_info, dht_map)

            dht_has_this_ip = False
            for info in dht_map.values():
                if node_id in info.keys():
                    dht_has_this_ip = True

            if not dht_has_this_ip or node_id not in dht_map[str(stage)]:
                print(f'Skip rebalance, node_id = {node_id}, stage = {stage}, dht_has_this_ip = {dht_has_this_ip}')
                return False

            print(f'stage={stage}, {node_id}, map={dht_map}')
            # assert qi == dht_map[str(stage)][node_id]['load'], f'{qi} != {dht_map[str(stage)][node_id]["load"]}'
        
            min_load_stages = get_min_load_stages(dht_map, lmin)
            # print(f'node stage = {stage}, ip:host = {node_id}, min_stages = {min_load_stages}, max_stage = {smax}')
            # if int(stage) == int(smin) and smax is not None and int(smax) != int(smin):
            if int(stage) in min_load_stages and int(stage) != int(smax) and len(dht_map[str(stage)]) > 1:
                # peers_max = dht_map.get(str(smax), {})
                print(f"Node {node_id}: migrating from stage {stage} to {smax}")
                self.node_info.set_stage(smax)
                return True
        except Exception as e:
            print(f"Node {self.node_info.id}: Error during rebalance: {e}")
            traceback.print_exc()
            return False
        
    def measure_load(self):
        return self.task_scheduler.running_tasks_count

