# !pip install heapdict
from heapdict import heapdict
import math

INF = math.inf

class MinPriorityQueue:
    def __init__(self):
        self.heap = heapdict()

    def add_or_update(self, node_id, key):
        """Добавляет или обновляет ключ вершины"""
        self.heap[node_id] = key

    def remove(self, node_id):
        """Удаляет вершину из очереди"""
        if node_id in self.heap:
            self.heap.pop(node_id)

    def pop(self):
        """Извлекает вершину с минимальным ключом"""
        if not self.heap:
            raise KeyError("pop from an empty priority queue")
        node_id, key = self.heap.popitem()
        return node_id, key

    def top_key(self):
        """Возвращает минимальный ключ без удаления"""
        if not self.heap:
            return INF
        _, key = self.heap.peekitem()
        return key

    def is_empty(self):
        return len(self.heap) == 0
