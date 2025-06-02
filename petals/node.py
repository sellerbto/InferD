import asyncio
from kademlia_client import DistributedHashTableServer
from aiohttp import web
import aiohttp
from task import *
from node_info import NodeInfo
from path_finder import PathFinder
from balance import Balancer
from task_scheduler import TaskScheduler

class Node:
    def __init__(self, 
            node_ip,
            node_port,
            num_stages: int,
            capacity: int,
            dht: DistributedHashTableServer,
            task_dht: DistributedHashTableServer,
            rebalance_period: int = 10):
        self.node_info = NodeInfo(id=f'{node_ip}:{node_port}',
                            ip=node_ip,
                            port=node_port,
                            num_stages=num_stages,
                            capacity=capacity,
                            rebalance_period=rebalance_period)
        self.dht = dht # stage -> url -> loading
        self.task_dht = task_dht # это только для тестирование swarm'а
        self.task_scheduler = TaskScheduler(self.node_info, self.dht, self.task_dht)
        self.balancer = Balancer(self.dht, self.node_info, self.task_scheduler, 5)
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

        self.completed_tasks = {} # task_id -> result (obj)


    async def rebalance_task(self):
        await asyncio.sleep(0.5)
        while True:
            await asyncio.sleep(10)
            await self.rebalance()

    async def change_stage(self, new_stage):
        record = await self.dht.get(str(self.node_info.stage)) or {}
        record.pop(self.node_info.id, None)
        await self.dht.set(str(self.node_info.stage), record)

        self.node_info.set_stage(new_stage)
        await self.task_scheduler.announce()

    async def rebalance(self):
        print('Rebalancing...')
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
    
    # async def handle_completed_task(self, request):
    #     data = await request.json()
    #     task_id = data.get('task_id')
    #     result_for_user = data.get('result_for_user')
    #     self.completed_tasks[task_id] = result_for_user


    async def post(self, url, payload):
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload) as resp:
                    data = await resp.json()
                    return data
            except Exception as e:
                print(f"Error sending task to {url}: {e}")
                return None

    
    async def run_task(self, task: Task):
        await self.task_scheduler.run_task(task)
        cur_stage = task.stage
        task_result = task.get_result()
        
        if cur_stage == 2:
            return {'result_for_user': task_result}
        
        next_node = await self.path_finder.find_best_node(stage=cur_stage + 1)
        if not next_node:
            return {'error': 'No next node available'}
        
        ip, port = next_node
        url = f"http://{ip}:{port}/nn_forward"
        payload = {
            'task_id': task.id,
            'stage': cur_stage + 1,
            'input_data': task_result,
        }

        print(f'Node {self.node_info.id} send to {url}, payload={payload}')
        response = await self.post(url, payload)
        print(f'Node {self.node_info.id} receives from {url}, response={response}')
        return response


    async def handle_nn_forward(self, request):
        # if not self._inited:
        #     await asyncio.sleep(3)  # чтобы успели все сервера инициализироваться в dht и не было пустых стейджей. если мы обратимся к пустому стейджу, то этот метод упадет
        #     self._inited = True

        data = await request.json()
        print(f'Data: {data}')
        stage = int(data['stage'])
        task = NNForwardTask(int(data['task_id']), stage, data['input_data'])

        if stage != self.node_info.stage:
            return web.json_response({
                'error': 'This node is not responsible for this task'
            })
        # ip, port = await self.path_finder.find_best_node(task.stage)
        
        # if port == self.node_info.port:
        #     task_result = await self.run_task(task)
        #     print(f'Task result = {task_result}')
        #     return web.json_response(task_result)

        task_result = await self.run_task(task)
        print(f'Task result = {task_result}')
        return web.json_response(task_result)


    async def start(self, initial_stage: int, rebalance=True):
        self.node_info.set_stage(initial_stage)
        if rebalance:
            self._rebalance_task = asyncio.create_task(self.rebalance_task())
            print("⚙️ rebalance_task created:", self._rebalance_task, flush=True)
        else:
            self._rebalance_task = None
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.node_info.ip, self.node_info.port)
        await site.start()
        print(f"Node {self.node_info.id}: HTTP API on port {self.node_info.ip}:{self.node_info.port}")


    async def stop(self):
        if self._rebalance_task:
            self._rebalance_task.cancel()
