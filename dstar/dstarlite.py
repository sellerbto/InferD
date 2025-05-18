import math
from priority_queue import MinPriorityQueue

INF = math.inf

class DStarLite:
    def __init__(self, node_layers, node_cost, edge_cost, passed_nodes):
        # Исходные данные
        self.node_layers = node_layers                  # список списков id узлов
        self.node_cost = node_cost                      # dict[node] -> float
        self.edge_cost = edge_cost                      # dict[(u,v)] -> float
        # Скомбинированные стоимости ребёр с учётом стоимости узла-приемника
        self.mod_edge = {
            (u, v): edge_cost[(u, v)] + node_cost[v]
            for (u, v) in edge_cost
        }
        # Словарь: node_id -> индекс слоя, где встречается
        self.layer_map = {}
        for idx, layer in enumerate(self.node_layers[:-1]):
            for n in layer:
                if n not in self.layer_map:
                    self.layer_map[n] = idx

        # Текущие пройденные узлы
        self.passed_nodes = passed_nodes
        self.start = (passed_nodes[-1], self.layer_map[passed_nodes[-1]])
        self.goal = (node_layers[0][0], len(node_layers) - 1)

        # Алгоритмические структуры
        self.g = {}
        self.rhs = {}
        self.U = MinPriorityQueue()

        # Инициализация g и rhs
        for idx, layer in enumerate(self.node_layers):
            for n in layer:
                state = (n, idx)
                self.g[state] = INF
                self.rhs[state] = INF
        # Для цели
        self.rhs[self.goal] = 0.0
        self.U.add_or_update(self.goal, 0.0)

    def successors(self, state):
        node, idx = state
        if idx < len(self.node_layers) - 1:
            return [(n, idx + 1) for n in self.node_layers[idx + 1]]
        return []

    def predecessors(self, state):
        node, idx = state
        if idx > 0:
            return [(n, idx - 1) for n in self.node_layers[idx - 1]]
        return []

    def update_vertex(self, u):
        if u != self.goal:
            self.rhs[u] = min(
                self.mod_edge.get((u[0], s[0]), INF) + self.g[s]
                for s in self.successors(u)
            )
        # Обновляем очередь
        self.U.add_or_update(u, min(self.g[u], self.rhs[u]))

    def compute_shortest_path(self):
        while (self.U.top_key() < self.rhs[self.start] or
               not (self.rhs[self.start] == self.g[self.start] != INF)):
            u, _ = self.U.pop()
            if self.g[u] > self.rhs[u]:
                # overconsistent
                self.g[u] = self.rhs[u]
                for p in self.predecessors(u):
                    self.update_vertex(p)
            if self.g[u] < self.rhs[u]:
                # underconsistent
                self.g[u] = INF
                self.update_vertex(u)
                for p in self.predecessors(u):
                    self.update_vertex(p)

    def update_edges(self, updates, passed_nodes):
        # Массовое обновление ребер: список (u, v, new_cost)
        self.start = (passed_nodes[-1], self.layer_map[passed_nodes[-1]])
        for edge, cost in updates.items():
            u, v = edge
            self.edge_cost[(u, v)] = cost
            self.mod_edge[(u, v)] = cost + self.node_cost[v]
        for u in set(e[0] for e, _ in updates.items()):
            self.update_vertex((u, self.layer_map[u]))

    def find_best_chain(self):
        self.compute_shortest_path()
        path = []
        current = self.start
        while current != self.goal:
            path.append(current[0])
            succs = self.successors(current)
            valid = [s for s in succs if self.mod_edge.get((current[0], s[0]), INF) + self.g[s] < INF]
            if not valid:
                break
            current = min(valid, key=lambda s: self.mod_edge[(current[0], s[0])] + self.g[s])
        path.append(self.goal[0])
        return path
