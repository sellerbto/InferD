import asyncio
from kademlia_client import DistributedHashTableServer
from aiohttp import web, ClientSession
from task import *
from config import port_shift
from node_info import NodeInfo
from path_finder import PathFinder
from balance import Balancer
from task_scheduler import TaskScheduler

class Node:
    def __init__(self, 
            node_port,
            num_stages: int,
            capacity: int,
            dht: DistributedHashTableServer,
            task_dht: DistributedHashTableServer,
            rebalance_period: int = 10):
        self.node_info = NodeInfo(id=f'127.0.0.1:{node_port}',
                            ip='127.0.0.1',
                            port=node_port,
                            num_stages=num_stages,
                            capacity=capacity,
                            rebalance_period=rebalance_period)
        self.dht = dht
        self.task_dht = task_dht
        self.task_scheduler = TaskScheduler(self.node_info, self.dht, self.task_dht)
        self.balancer = Balancer(self.dht, self.node_info, self.task_scheduler, 2)
        self.path_finder = PathFinder(self.dht,
                                    self.node_info,
                                    self.balancer)
        self._rebalance_task = None

        self.app = web.Application()
        self.app.add_routes([
            web.post('/nn_forward', self.handle_nn_forward),
            web.post('/reassign',  self.handle_reassign),
            ])
        self.node_addresses = self.dht.bootstrap_nodes
        self._inited = False


    async def rebalance_task(self):
        await asyncio.sleep(0.5)
        while True:
            await asyncio.sleep(1)
            await self.rebalance()

    async def change_stage(self, new_stage):
        record = await self.dht.get(str(self.node_info.stage)) or {}
        record.pop(self.node_info.id, None)
        await self.dht.set(str(self.node_info.stage), record)

        self.node_info.set_stage(new_stage)
        await self.task_scheduler.announce()

    async def rebalance(self):
        return await self.balancer.rebalance()
    
    async def handle_reassign(self, request):
        """
        {"to": <int>}
        """
        data = await request.json()
        new_stage = data.get('to')
        await self.change_stage(new_stage)
        print(f'Node {self.node_info.id} changed stage from {self.node_info.stage} to {new_stage}, {type(self.node_info.stage)}, {type(new_stage)}')
        return web.json_response({
            'node_info.id': self.node_info.id,
            'new_stage': self.node_info.stage,
            'status': 'reassigned'
        })   

    async def handle_nn_forward(self, request):
        if not self._inited:
            await asyncio.sleep(3)  # чтобы успели все сервера инициализироваться в dht и не было пустых стейджей. если мы обратимся к пустому стейджу, то этот метод упадет
            self._inited = True

        data = await request.json()
        task = NNForwardTask(int(data['stage']), data['input_data'])
        ip, port = await self.path_finder.find_best_node(task.stage)
        if port == self.node_info.port:
            asyncio.create_task(self.task_scheduler.run_task(task))
            return web.json_response({'node_info.id': self.node_info.id, 'processed': True})
        http_port = port
        url = f"http://{ip}:{http_port}/nn_forward"

        async with ClientSession() as session:
            async with session.post(url, json=data) as resp:
                forward_result = await resp.json()
            print(f'Send {task} to host {ip} on port {port}')
        return web.json_response(forward_result)


    async def start(self, initial_stage: int, rebalance=True):
        self.node_info.set_stage(initial_stage)
        self._rebalance_task = asyncio.create_task(self.rebalance_task()) if rebalance else None
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.dht.port + port_shift)
        await site.start()
        print(f"Node {self.node_info.id}: HTTP API on port {self.dht.port + port_shift}")


    async def stop(self):
        if self._rebalance_task:
            self._rebalance_task.cancel()
