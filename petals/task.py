import torch.nn as nn
from typing import List, Optional
import torch
from transformers import AutoTokenizer
from partitioned_models import PartitionedQwen2

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
    def __init__(self, id, stage, input_data):
        self.id = id
        self.stage = stage
        self.input_data = input_data
        self.result = None
        self.runned = False
        self.state = input_data

    def __repr__(self):
        return f'NNForwardTask({self.stage=}, {self.input_data=})'
    
    def run(self):
        print(f'Task runned: {self.stage=}, state: {self.state} -> {self.state+1}')
        self.state += 1
        # self.runned = True
    
    def get_result(self):
        return self.state


# class MLPTask(Task):
#     def __init__(self, id: int, stage: int, input_data):
#         self.id = id
#         self.stage = stage
#         self.model = PartitionedMLP(stage)
#         self.input_data = input_data
#         self.result = None

#     def run(self):
#         input_tensor = torch.tensor(self.input_data, dtype=torch.float32)
#         self.result = self.model.forward(input_tensor).detach().numpy().tolist()

#     def get_result(self):
#         return self.result

class QwenTask(Task):
    def __init__(self, id: int, stage: int, input_data):

        self.id = id
        self.stage = stage
        self.input_data = input_data
        self.model = PartitionedQwen2(stage)
        self.result = None

    def run(self):
        if self.stage == 0:
            # Ожидаем input_data как prompt:str
            prompt = self.input_data
            # Вызываем forward только для stage=0 → получаем hidden_states
            hidden1 = self.model.forward(prompt)
            # Сохраняем в result
            self.result = hidden1

        elif self.stage == 1:
            # Ожидаем input_data как dict с hidden_states и attention_mask
            hidden_states = self.model.forward(self.input_data)
            self.result = hidden_states

        else:  # self.stage == LAST_STAGE
            # Ожидаем input_data как prompt:str
            hidden_states = self.model.forward(self.input_data)
            self.result = hidden_states

    def get_result(self):
        return self.result
