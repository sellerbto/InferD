from task import Task, QwenTask
import asyncio
from node_info import NodeInfo

class TaskScheduler:
    def __init__(self,
                node_info: NodeInfo,
                dht):
        self.node_info = node_info
        self.dht = dht
        self.running_tasks_count = 0

    def measure_load(self):
        return self.running_tasks_count

    async def run_task(self, task: QwenTask):
        cur_stage = self.node_info.stage
        task.run()
        self.running_tasks_count += 1
        await self.announce()
        asyncio.create_task(self._finish_task(cur_stage))


    async def _finish_task(self, cur_stage: int):
        # await asyncio.sleep(10)
        self.running_tasks_count -= 1
        await self.announce()

    async def announce(self):
        try:
            load = self.measure_load()
            record = await self.dht.get(str(self.node_info.stage)) or {}
            record[self.node_info.id] = {'load': load, 'cap': self.node_info.capacity}
            await self.dht.set(str(self.node_info.stage), record)
        except Exception as e:
            print(f"Node {self.node_info.id}: Failed to announce - {e}")
