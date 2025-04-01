from queue import PriorityQueue
from typing import Set
import time
from server import Server


class StochasticWiring:
    def __init__(self, number_of_pipeline_stages, active_servers: Set[Server], smoothing_parameter, initial_priority):
        self.number_of_pipeline_stages = number_of_pipeline_stages
        self.active_servers = active_servers
        self.smoothing_parameter = smoothing_parameter
        self.initial_priority = initial_priority
        self.ema = {}
        self.queues = [PriorityQueue() for _ in range(number_of_pipeline_stages)]


    def get_blocks_served_by_(self, server):
        return []


    def add_server_(self, server):
        self.active_servers.add(server)
        self.ema[server] = self.initial_priority
        for i in self.get_blocks_served_by_(server):
            self.queues[i].put(server, priority=self.initial_priority)


    def ban_server_(self, server):
        self.active_servers.remove(server)
        self.ema.pop(server)
        for i in self.get_blocks_served_by_(server):
            self.queues[i].put(server, priority=float('inf'))


    def choose_server(self, layer_index):
        server, priority = self.queues[layer_index].get()
        new_priority = priority + self.ema[server]
        for j in self.get_blocks_served_by_(server):
            self.queues[j].put(server, priority=new_priority)
        return server


    def forward(self, inputs):
        layer_index = 0
        while layer_index < self.number_of_pipeline_stages:
            server = self.choose_server(layer_index)
            t = time.time()
            try:
                inputs = server.forward(inputs)
                layer_index += 1
                delta_t = time.time() - t
                self.ema[server] = self.smoothing_parameter * delta_t + (1 - self.smoothing_parameter) * self.ema[server]
            except TimeoutError:
                self.ban_server_(server)
        return inputs

    def __call__(self, inputs):
        return self.forward(inputs)
