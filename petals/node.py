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
        self.app.add_routes([web.post('/nn_forward', self.handle_nn_forward)])
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

    async def rebalance(self):
        await asyncio.sleep(1)
        while True:
            try:
                await asyncio.sleep(self.rebalance_period)
                qi = await self.measure_load()
                print(f'LOAD = {qi}')
                record = await self.dht.get(str(self.stage)) or {}
                record[self.node_id] = {'load': qi, 'cap': self.capacity}
                await self.dht.set(str(self.stage), record)

                smin = smax = None
                lmin = float('inf')
                lmax = float('-inf')

                dht_map = await self.dht.get_all() or {}
                print(dht_map)
                assert qi == (await self.dht.get(str(self.stage)))[self.node_id]['load'], f'{qi} != {dht_map[self.node_id]["load"]}'

                print(f'node stage = {self.stage}, peers = {dht_map.items()}')
                for stage_str, peers in dht_map.items():
                    stage_i = int(stage_str)
                    L = sum(p['load'] for p in peers.values())
                    print(f'node stage = {self.stage}, load = {L}, stage number = {stage_i}')
                    if L > lmax:
                        lmax, smax = L, stage_i
                    if L < lmin:
                        lmin, smin = L, stage_i

                print(f'node stage = {self.stage}, min_stage = {smin}, max_stage = {smax}, dht = {await self.dht.get_all()}')
                if self.stage == smin and smax is not None and smax != smin:
                    peers_max = dht_map.get(str(smax), {})
                    imin, qmin = min(peers_max.items(), key=lambda item: item[1]['load'])
                    if imin == self.node_id:
                        print(f"Node {self.node_id}: migrating from stage {smin} to {smax}")
                        await self.ban_self()
                        await self.set_stage(smax)
                        await self.announce()
            except Exception as e:
                print(f"Node {self.node_id}: Error during rebalance: {e}")
                traceback.print_exc()

    async def find_best_node(self, stage: int) -> int:
        # todo: D^* algorithm...
        record = await self.dht.get(str(stage)) or {}
        if not record:
            raise RuntimeError(f"No peers available for stage {stage}")
        else:
            print(f'Found {stage}: {record}')
        ip_port, _ = min(record.items(), key=lambda x: x[1]['load'])
        await asyncio.sleep(1)
        return parse_ip_port(ip_port)

    async def run_task(self, task: Task):
        print(f'RUN TASK')
        cur_stage = self.stage

        # увеличить счётчик задач в task_dht[cur_stage]
        td_rec = await self.task_dht.get(str(cur_stage)) or {}
        td_count = td_rec.get(self.node_id, 0) + 1
        td_rec[self.node_id] = td_count
        await self.task_dht.set(str(cur_stage), td_rec)

        self.running_tasks_count += 1
        await asyncio.sleep(1)
        self.running_tasks_count -= 1

        td_rec = await self.task_dht.get(str(cur_stage)) or {}
        td_count = td_rec.get(self.node_id, 1) - 1
        if td_count > 0:
            td_rec[self.node_id] = td_count
        else:
            td_rec.pop(self.node_id, None)
        await self.task_dht.set(str(cur_stage), td_rec)

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
        self._rebalance_task = asyncio.create_task(self.rebalance())
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.dht.port + port_shift)
        await site.start()
        print(f"Node {self.node_id}: HTTP API on port {self.dht.port + port_shift}")

    async def stop(self):
        if self._rebalance_task:
            self._rebalance_task.cancel()
