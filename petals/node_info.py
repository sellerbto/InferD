class NodeInfo:
    def __init__(self, 
                id, 
                ip,
                port,
                name,
                model_name,
                num_stages,
                capacity,
                rebalance_period,
                stage=None):
        self.id = id
        self.ip = ip
        self.port = port
        self.name = name
        self.model_name = model_name
        self.stage = stage
        self.num_stages = num_stages
        self.capacity = capacity
        self.queue_size = 0
        self.rebalance_period = rebalance_period

    def set_stage(self, new_stage: int):
        # if self.stage == new_stage:
        #     raise Exception(f'')
        # self.stage = new_stage
        # print(f"Node {self.id}: loaded stage {new_stage}")
        pass