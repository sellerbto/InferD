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


class QwenTask(Task):
    def __init__(self, id: int, model, stage: int, input_data):
        self.id = id
        self.stage = stage
        self.input_data = input_data
        self.model = model
        self.result = None

    def run(self):
        out = self.model.forward(self.input_data)
        self.result = out

    def get_result(self):
        return self.result

    def __repr__(self):
        return f"<QwenTask id={self.id} stage={self.stage}>"
