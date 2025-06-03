import asyncio
from kademlia_client import DistributedHashTableServer
from aiohttp import web
import aiohttp

from task import QwenTask
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
        self.node_info = NodeInfo(
            id=f'{node_ip}:{node_port}',
            ip=node_ip,
            port=node_port,
            num_stages=num_stages,
            capacity=capacity,
            rebalance_period=rebalance_period
        )

        self.dht = dht
        self.task_dht = task_dht
        self.task_scheduler = TaskScheduler(self.node_info, self.dht, self.task_dht)
        self.balancer = Balancer(self.dht, self.node_info, self.task_scheduler, 5)
        self.path_finder = PathFinder(self.dht, self.node_info, self.balancer)
        self._rebalance_task = None

        # Настраиваем HTTP-сервер
        self.app = web.Application()
        self.app.add_routes([
            web.post('/nn_forward', self.handle_nn_forward),
            web.post('/reassign',  self.handle_reassign),
        ])

        self.node_addresses = self.dht.bootstrap_nodes
        self._inited = False

        # Для Stage 0: храним объект QwenTask по task_id,
        # чтобы накапливать hidden_meta и generated_ids
        self.active_tasks = {}

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

    async def post(self, url, payload):
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload) as resp:
                    return await resp.json()
            except Exception as e:
                print(f"Error sending to {url}: {e}")
                return None

    
    async def run_task(self, task: QwenTask):
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
        """
        Ожидаем JSON:
          {
            "task_id": <любой>,
            "stage": <0,1 или 2>,
            "input_data": <строка или dict>
          }
        """
        data = await request.json()
        task_id = data["task_id"]
        stage = int(data["stage"])
        payload = data["input_data"]

        # Если запрос не на этот stage — сразу ошибка
        if stage != self.node_info.stage:
            return web.json_response({
                'error': f'Node at stage {self.node_info.stage} cannot handle stage {stage}'
            })

        from partitioned_models import LAST_STAGE

        # === STAGE 0 ===
        if stage == 0:
            prompt = payload  # строка
            generated_text = ""
            gen_ids = []

            # Первый шаг: run_task(stage=0, prompt) автоматически пошлет chain→1→2
            task0 = QwenTask(task_id, 0, prompt)
            resp0 = await self.run_task(task0)
            # resp0 == {"next_token_str": строка, "generated_ids": [...]}
            nxt = resp0["next_token_str"]
            gen_ids = resp0["generated_ids"]
            if nxt == "":
                # сразу EOS
                return web.json_response({'result_for_user': ""})
            generated_text += nxt

            # Далее — повторяем до 49 раз:
            for _ in range(5):
                # a) пересчет Stage0 с уже имеющимися gen_ids:
                task0 = QwenTask(task_id, 0, {"generated_ids": gen_ids})
                resp0 = await self.run_task(task0)
                nxt = resp0["next_token_str"]
                gen_ids = resp0["generated_ids"]
                if nxt == "" or nxt == task0.model.tokenizer.eos_token:
                    break
                generated_text += nxt

            return web.json_response({'result_for_user': generated_text})

        # === STAGE 1 ===
        if stage == 1:
            # payload == {"hidden_meta": …, "generated_ids": […]}
            task1 = QwenTask(task_id, 1, payload)
            resp = await self.run_task(task1)
            # resp == {"next_token_str": строка, "generated_ids": […]}
            return web.json_response({
                'next_token_str': resp["next_token_str"],
                'generated_ids': resp["generated_ids"]
            })

        # === STAGE 2 ===
        if stage == LAST_STAGE:
            # payload == {"hidden_meta": …, "generated_ids": […]}
            task2 = QwenTask(task_id, LAST_STAGE, payload)
            resp = await self.run_task(task2)
            # resp == {'result_for_user': {"next_token_str":…, "generated_ids": […]}}
            return web.json_response(resp["result_for_user"])

        return web.json_response({'error': 'Invalid stage'})


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
        print(f"Node {self.node_info.id}: HTTP API on {self.node_info.ip}:{self.node_info.port}")

    async def stop(self):
        if self._rebalance_task:
            self._rebalance_task.cancel()
