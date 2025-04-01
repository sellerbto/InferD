from node import Node
from kademlia_client import DistributedHashTableServer
from typing import List
import asyncio
import config

async def fetch_nodes_info(nodes: List[Node], dht: DistributedHashTableServer, verbose=False):
    stage_nums = config.STAGE_NUMS
    node_id_to_stage = {}
    stage_to_load = {}
    if verbose:
        print('====================')
    for stage_num in range(stage_nums):
       cur_stage_info = await dht.get(stage_num)
       stage_to_load[stage_num] = cur_stage_info
       if verbose:
           print(f'cur stage = {stage_num}, info = {cur_stage_info}')
       for node_id, loading in cur_stage_info.items():
           node_id_to_stage[node_id] = stage_num

    for node in nodes:
        if node.id in node_id_to_stage:
            node.stage_number = node_id_to_stage[node.id]
        else:
            print(f'Node {node.id} not found in stage info')
            stage_to_load[node.stage_number][node.id] = node.get_local_queue_size()

    for stage_num, nodes_info in stage_to_load.items():
        await dht.set(stage_num, nodes_info)

    return stage_to_load

async def rebalance(node: Node, dht: DistributedHashTableServer, stages=3, T=1):
    prefix = f'Node {node.id} (stage {node.stage_number})'
    while True:
        await fetch_nodes_info([node], dht)
        # print(prefix)
        # print(f' Node =========== {node.id}, stage number = {node.stage_number}')
        # print(f'{prefix} sleeps')
        await asyncio.sleep(T)
        queue_size = node.get_local_queue_size()
        loading_info : dict
        loading_info = await dht.get(node.stage_number)

        # print(f'{prefix} loading: {loading_info}')
        # print(sorted(list(loading_info.items())))

        if loading_info is None:
            loading_info = {}
        loading_info[node.id] = queue_size
        await dht.set(node.stage_number, loading_info)

        stage_loads = {} # key - stage number, value - total queue size
        for s in range(stages):
            stage_loads[s] = 0
            data = await dht.get(s)
            if data:
                for _, q in data.items():
                    stage_loads[s] += q

        # print(f'{prefix}, stage loads: {sorted(list(stage_loads.items()))}')

        min_stage = min(range(stages), key=lambda x: stage_loads.get(x, 0))
        max_stage = max(range(stages), key=lambda x: stage_loads.get(x, 0))

        # print(min_stage, max_stage)

        if int(node.stage_number) == int(min_stage):
            min_peer, min_queue = None, float("inf")

            if loading_info:
                for peer_id, q in loading_info.items():
                    if q < min_queue:
                        min_peer, min_queue = peer_id, q
            # print('min_peer', min_peer, 'min_queue', min_queue, 'node_id', node.id)

            if int(min_peer) == int(node.id):
                # print(f"Node {node.id} migrating from stage {node.stage_number} to {max_stage}")
                cur_stage_loading = await dht.get(node.stage_number)
                # print(f'cur_stage_loading: {cur_stage_loading}')
                if cur_stage_loading and node.id in cur_stage_loading and len(cur_stage_loading) > 1:
                    del cur_stage_loading[node.id]
                    # print(f'cur_stage_loading after delete {node.id}: {cur_stage_loading}')
                await dht.set(node.stage_number, cur_stage_loading)

                node.stage_number = max_stage
                max_stage_loading : dict = await dht.get(max_stage)
                if max_stage_loading is None:
                    max_stage_loading = {}
                max_stage_loading[node.id] = queue_size
                await dht.set(max_stage, max_stage_loading)
