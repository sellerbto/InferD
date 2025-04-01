import random

class Node:
    def __init__(self, id, stage_number, queue_size, capacity=10):
        self.stage_number = stage_number
        self.id = id
        self.queue_size = queue_size
        self.capacity = capacity
        self.load = 0

    def is_available(self):
        return self.load < self.capacity

    def get_local_queue_size(self):
        return self.queue_size

    def get_compute_capacity(self):
        return self.capacity

    def assign_task(self, task):
        if self.is_available():
            self.load += task.size
            return True
        return False

    def get_metrics(self):
        raise NotImplementedError

    def release_task(self, task):
        self.load -= task.size
