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
            hidden1 = self.model.forward(prompt=prompt)
            # Сохраняем в result
            self.result = hidden1

        elif self.stage == 1:
            # Ожидаем input_data как dict с hidden_states и attention_mask
            hidden_states = self.input_data.get("hidden_states", None)
            attention_mask = self.input_data.get("attention_mask", None)
            if hidden_states is None or attention_mask is None:
                raise ValueError("Для stage=1 нужно передать {'hidden_states': Tensor, 'attention_mask': Tensor}")
            # Вызываем forward для stage=1 → получаем hidden2
            hidden2 = self.model.forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask
            )
            self.result = hidden2

        else:  # self.stage == LAST_STAGE
            # Ожидаем input_data как prompt:str
            prompt = self.input_data
            # Вызываем forward для последнего этапа → получаем весь сгенерированный текст
            full_text = self.model.forward(
                prompt=prompt,
                max_new_tokens=10,
                do_sample=True,
                temperature=1.0,
                top_k=50,
                top_p=0.9
            )
            self.result = full_text

    def get_result(self):
        return self.result
