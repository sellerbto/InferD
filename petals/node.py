import asyncio
from kademlia_client import DistributedHashTableServer
from aiohttp import web, ClientSession
from task import *
from config import port_shift
import traceback
from utils import parse_ip_port

class Node:
    def __init__(self, 
            node_port,
            num_stages: int, 
            capacity: int,
            dht: DistributedHashTableServer,
            task_dht: DistributedHashTableServer,
            rebalance_period: int = 10):
        self.port = node_port
        self.node_id = f'127.0.0.1:{node_port}'
        self.num_stages = num_stages
        self.capacity = capacity
        self.stage = None
        self.queue_size = 0
        self.rebalance_period = rebalance_period
        self.dht = dht
        self.task_dht = task_dht
        self._rebalance_task = None

        self.running_tasks_count = 0

        self.app = web.Application()
        self.app.add_routes([
            web.post('/nn_forward', self.handle_nn_forward),
            web.post('/reassign',  self.handle_reassign),
            ])
        self.node_addresses = self.dht.bootstrap_nodes
        self._inited = False

    async def set_stage(self, stage_index: int):
        self.stage = stage_index
        print(f"Node {self.node_id}: loaded stage {stage_index}")

    async def measure_load(self) -> int:
        return self.running_tasks_count

    async def announce(self):
        try:
            load = await self.measure_load()
            record = await self.dht.get(str(self.stage)) or {}
            record[self.node_id] = {'load': load, 'cap': self.capacity}
            await self.dht.set(str(self.stage), record)
        except Exception as e:
            print(f"Node {self.node_id}: Failed to announce - {e}")

    async def ban_self(self):
        record = await self.dht.get(str(self.stage)) or {}
        record.pop(self.node_id, None)
        await self.dht.set(str(self.stage), record)

    async def rebalance_task(self):
        await asyncio.sleep(0.5)
        while True:
            await asyncio.sleep(1)
            await self.rebalance()

    async def change_stage(self, new_stage):
        await self.ban_self()
        await self.set_stage(new_stage)
        await self.announce()

    async def min_max_load_stage(self, dht_map):
        lmin = float('inf')
        lmax = float('-inf')
        smin = None
        smax = None
        for stage_str, peers in dht_map.items():
            stage_i = int(stage_str)
            L = sum(p['load'] for p in peers.values())
            print(f'node stage = {self.stage}, ip:host = {self.node_id}, load = {L}, stage number = {stage_i}')
            if L > lmax:
                lmax, smax = L, stage_i
            if L < lmin:
                lmin, smin = L, stage_i
        return lmin, lmax, smin, smax
    
    def get_min_load_stages(self, dht_map, lmin):
        min_load_stages = set()
        for stage, info in dht_map.items():
            if not info:
                continue
            if sum(p['load'] for p in info.values()) == lmin:
                min_load_stages.add(int(stage))
        return min_load_stages

    async def rebalance(self):
        try:
            await asyncio.sleep(self.rebalance_period)
            qi = await self.measure_load()
            # print(f'LOAD = {qi}')
            record = await self.dht.get(str(self.stage)) or {}
            record[self.node_id] = {'load': qi, 'cap': self.capacity}
            await self.dht.set(str(self.stage), record)

            dht_map = await self.dht.get_all()
            lmin, lmax, smin, smax = await self.min_max_load_stage(dht_map)

            dht_has_this_ip = False
            for info in dht_map.values():
                if self.node_id in info.keys():
                    dht_has_this_ip = True

            if not dht_has_this_ip or self.node_id not in dht_map[str(self.stage)]:
                print(f'Skip rebalance, node_id = {self.node_id}, stage = {self.stage}, dht_has_this_ip = {dht_has_this_ip}')
                return False

            print(f'stage={self.stage}, {self.node_id}, map={dht_map}')
            # assert qi == dht_map[str(self.stage)][self.node_id]['load'], f'{qi} != {dht_map[str(self.stage)][self.node_id]["load"]}'
        
            min_load_stages = self.get_min_load_stages(dht_map, lmin)
            print(f'node stage = {self.stage}, ip:host = {self.node_id}, min_stages = {min_load_stages}, max_stage = {smax}')
            # if int(self.stage) == int(smin) and smax is not None and int(smax) != int(smin):
            if int(self.stage) in min_load_stages and int(self.stage) != int(smax):
                # peers_max = dht_map.get(str(smax), {})
                print(f"Node {self.node_id}: migrating from stage {self.stage} to {smax}")
                await self.change_stage(smax)
                return True
        except Exception as e:
            print(f"Node {self.node_id}: Error during rebalance: {e}")
            traceback.print_exc()
            return False


    async def find_best_node(self, stage: int, retry: int = 3) -> int:
        # todo: D^* algorithm...
        if retry == 0:
            print(f'Try to reassign node')
            dht_map = await self.dht.get_all() 
            lmin, lmax, smin, smax = await self.min_max_load_stage(dht_map)
            print(lmin, lmax, smin, smax)
    
            potential_nodes_to_reassign = dht_map[str(smin)].keys()
            print(f'potential_nodes_to_reassign = {potential_nodes_to_reassign}')
            ip, port, old_stage = None, None, None

            min_load_stages = self.get_min_load_stages(dht_map, lmin)
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
            print(f'No peers available for stage {stage}. Try to rebalance this node with stage = {self.stage} to stage {stage}')
            # надо какую-то ноду назначить на этот стейдж, сейчас у текущей ноды берется rebalance в надежде что эта нода перестроится на пустой стедж
            # raise RuntimeError(f"No peers available for stage {stage}") 
            old_stage = self.stage
            rebalanced = await self.rebalance() 
            rebalance_verbose = f'stage changed from {old_stage} to {self.stage}' if rebalanced else 'nothing changed'
            print(f'Rebalancing result: {rebalance_verbose}')
            print(f'Retry')
            return await self.find_best_node(stage, retry-1)
        else:
            print(f'Found {stage}: {record}')
        ip_port, _ = min(record.items(), key=lambda x: x[1]['load'])
        # await asyncio.sleep(1)
        return parse_ip_port(ip_port)
    
    async def handle_reassign(self, request):
        """
        {"to": <int>}
        """
        data = await request.json()
        new_stage = data.get('to')
        await self.change_stage(new_stage)
        print(f'Node {self.node_id} changed stage from {self.stage} to {new_stage}, {type(self.stage)}, {type(new_stage)}')
        return web.json_response({
            'node_id': self.node_id,
            'new_stage': self.stage,
            'status': 'reassigned'
        })

    async def reassign_node(self, ip: str, http_port: int, to_stage: int):
        url = f'http://{ip}:{http_port}/reassign'
        async with ClientSession() as session:
            async with session.post(url, json={'to': to_stage}) as resp:
                return await resp.json()    

    async def run_task(self, task: Task):
        cur_stage = self.stage
        td_rec = await self.task_dht.get(cur_stage) or {}
        td_rec[self.node_id] = td_rec.get(self.node_id, 0) + 1
        await self.task_dht.set(cur_stage, td_rec)
        self.running_tasks_count += 1
        await self.announce()
        asyncio.create_task(self._finish_task(cur_stage))


    async def _finish_task(self, cur_stage: int):
        await asyncio.sleep(10)
        self.running_tasks_count -= 1
        await self.announce()
        td_rec = await self.task_dht.get(cur_stage) or {}
        if td_rec.get(self.node_id, 0) > 1:
            td_rec[self.node_id] -= 1
        else:
            td_rec.pop(self.node_id, None)
        await self.task_dht.set(cur_stage, td_rec)


    async def handle_nn_forward(self, request):
        if not self._inited:
            await asyncio.sleep(3)  # чтобы успели все сервера инициализироваться в dht и не было пустых стейджей. если мы обратимся к пустому стейджу, то этот метод упадет
            self._inited = True

        data = await request.json()
        task = NNForwardTask(int(data['stage']), data['input_data'])
        ip, port = await self.find_best_node(task.stage)
        if port == self.port:
            asyncio.create_task(self.run_task(task))
            return web.json_response({'node_id': self.node_id, 'processed': True})
        http_port = port
        url = f"http://{ip}:{http_port}/nn_forward"

        async with ClientSession() as session:
            async with session.post(url, json=data) as resp:
                forward_result = await resp.json()
            print(f'Send {task} to host {ip} on port {port}')
        return web.json_response(forward_result)


    async def start(self, initial_stage: int):
        await self.set_stage(initial_stage)
        self._rebalance_task = asyncio.create_task(self.rebalance_task())
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.dht.port + port_shift)
        await site.start()
        print(f"Node {self.node_id}: HTTP API on port {self.dht.port + port_shift}")


    async def stop(self):
        if self._rebalance_task:
            self._rebalance_task.cancel()
