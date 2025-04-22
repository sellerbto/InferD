import torch.nn as nn
from typing import List, Optional


class Task:
    def __init__(self, input_data):
        self.input_data = input_data
        self.output_data = None

    def set_result(self, result):
        if self.output_data:
            raise AttributeError('Output also setted!')
        self.output_data = result

    def get_result(self):
        return self.output_data
    
    def get_stage_id(self) -> str:
        raise NotImplementedError
    

    

class UserTask(Task):
    pass

class NNForwardTask(Task):
    def __init__(self, stage, input_data):
        self.stage = stage
        self.input_data = input_data

    def __repr__(self):
        return f'NNForwardTask({self.stage}=, {self.input_data=})'
    


class TaskQueue:
    def __init__(self):
        self.queue: List[Task] = []

    def add_task(self, task: Task):
        self.queue.append(task)

    def pop_task(self) -> Optional[Task]:
        if self.queue:
            return self.queue.pop(0)
        return None

    def peek(self) -> Optional[Task]:
        if self.queue:
            return self.queue[0]
        return None

    def get_all_stage_ids(self) -> List[str]:
        return [task.get_stage_id() for task in self.queue]

    def get_task_by_stage_id(self, stage_id: str) -> Optional[Task]:
        for task in self.queue:
            if task.get_stage_id() == stage_id:
                return task
        return None

    def remove_task_by_stage_id(self, stage_id: str) -> bool:
        for i, task in enumerate(self.queue):
            if task.get_stage_id() == stage_id:
                del self.queue[i]
                return True
        return False

    def __len__(self):
        return len(self.queue)
    