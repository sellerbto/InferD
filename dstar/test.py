from dstarlite import DStarLite

node_layers = [[0], [1,2,3], [4,5], [0]]
node_cost = {
    0: 0,
    1: 3,
    2: 1, 
    3: 3,
    4: 2,
    5: 6
}
edge_cost = {
    (0, 1): 3,
    (0, 2): 1,
    (0, 3): 2,

    (1, 4): 7,
    (1, 5): 2,
    (2, 4): 4,
    (2, 5): 2,
    (3, 4): 4,
    (3, 5): 5,

    (5, 0): 2,
    (4, 0): 3
}
passed_nodes = [0]

algo = DStarLite(node_layers, node_cost, edge_cost, passed_nodes)
print(algo.find_best_chain()) # correct answer: [0,2,4,0]

# a few moments later

updated_edges = {
    (2, 4): 2,
    (2, 5): 3,
    (5, 0): 2,
    (4, 0): 11
}
passed_nodes = [0, 2]

algo.update_edges(updated_edges, passed_nodes)
# insert passed_nodes in algo. How?
print(algo.find_best_chain()) # correct answer: [2,5,0]


