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


class QwenTask(Task):
    def __init__(self, id: int, model: PartitionedQwen2, stage: int, input_data):
        self.id = id
        self.stage = stage
        self.input_data = input_data
        self.model = model
        self.result = None

    def run(self):
        out = self.model.forward(self.input_data)
        if "hidden_meta" in out and "generated_ids" not in out:
            out["generated_ids"] = self.input_data.get("generated_ids", [])
        self.result = out

    def get_result(self):
        return self.result

    def __repr__(self):
        return f"<QwenTask id={self.id} stage={self.stage}>"