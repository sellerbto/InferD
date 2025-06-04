import torch.nn as nn
from typing import List, Optional
import torch
from transformers import AutoTokenizer
from partitioned_models import PartitionedQwen2, LAST_STAGE

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


class QwenTask(Task):
    def __init__(self, id: int, stage: int, input_data):
        self.id = id
        self.stage = stage
        self.input_data = input_data   # либо str (для stage=0), либо dict
        self.model = PartitionedQwen2(stage)
        self.result = None

    def run(self):
        if self.stage == 0:
            if isinstance(self.input_data, str):
                out = self.model.forward({"prompt": self.input_data})
            else:
                out = self.model.forward({"generated_ids": self.input_data["generated_ids"]})
            # out == {"hidden_meta": ..., "generated_ids": [...]}
            self.result = out

        elif self.stage == 1:
            # input_data = {"hidden_meta": …, "generated_ids": […]} 
            hm = self.input_data["hidden_meta"]
            gen_ids = self.input_data["generated_ids"]
            out = self.model.forward({"hidden_meta": hm})
            # out == {"hidden_meta": …} для Stage 2
            self.result = {
                "hidden_meta": out["hidden_meta"],
                "generated_ids": gen_ids
            }

        # === STAGE 2 ===
        elif self.stage == LAST_STAGE:
            # ожидаем input_data = {"hidden_meta": …, "generated_ids": […]} 
            hm = self.input_data["hidden_meta"]
            gen_ids = self.input_data["generated_ids"]
            out = self.model.forward({"hidden_meta": hm, "generated_ids": gen_ids})
            # out == {"next_token_str": ..., "generated_ids": [...]}
            self.result = out

        else:
            raise RuntimeError(f"Invalid stage {self.stage} in QwenTask")

    def get_result(self):
        return self.result