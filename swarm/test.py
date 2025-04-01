import asyncio
from multiprocessing import Process
from node import Node
from rebalancing import rebalance
from kademlia_client import DistributedHashTableServer
import time
import asyncio
from rebalancing import fetch_nodes_info
from config import *

nodes = [
    Node(0, 0, 1),
    Node(1, 0, 2),
    Node(2, 1, 1),
    Node(3, 1, 50),
    Node(4, 2, 5),
    Node(5, 2, 0),
    Node(6, 2, 1),
    Node(7, 2, 2)
]

id_to_node = {node.id: node for node in nodes}
NUM_NODES = len(nodes)
ports = [6000 + i for i in range(NUM_NODES)]
bootstrap_nodes = [("127.0.0.1", port) for port in ports]


def start_node(node_id, port, bootstrap_nodes):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    dht = DistributedHashTableServer(port, bootstrap_nodes)

    async def run():
        await dht.start()
        print(f"🟢 Узел {node_id} запущен на порту {port}")
        await rebalance(nodes[node_id], dht)

    loop.run_until_complete(run())
    loop.run_forever()

def check_nodes_status():
    loop = asyncio.new_event_loop()

    async def get_capacity_loading_ratio():
        dht = DistributedHashTableServer(ports[-1]+1, bootstrap_nodes)
        await dht.start()
        while True:
            stage_to_capacity = {}
            stage_to_nodes = {}
            stage_to_load_sum = {}
            time.sleep(2)

            stage_to_load = await fetch_nodes_info(nodes, dht, verbose=True)
            load_sum = sum([sum(stage_to_load[stage].values())for stage in stage_to_load])
            print('ASSERT ',load_sum, sum([node.get_local_queue_size() for node in nodes]))
            for stage_num in range(STAGE_NUMS):
                stage_to_load_sum[stage_num] = stage_to_load_sum.get(stage_num, 0) + sum(stage_to_load[stage_num].values())
                stage_to_capacity[stage_num] = stage_to_capacity.get(stage_num, 0) + sum(id_to_node[node_id].get_compute_capacity() for node_id in stage_to_load[stage_num].keys())
                stage_to_nodes[stage_num] = set(stage_to_load[stage_num].keys())

            for stage in range(STAGE_NUMS):
                print(f'stage={stage}, loading={stage_to_load_sum[stage]}, capacity={stage_to_capacity[stage]}, nodes={stage_to_nodes[stage]}')

    loop.run_until_complete(get_capacity_loading_ratio())
    loop.run_forever()


if __name__ == "__main__":
    processes = []
    for i in range(NUM_NODES):
        time.sleep(0.5)
        p = Process(target=start_node, args=(i, ports[i], bootstrap_nodes))
        p.start()
        processes.append(p)

    p = Process(target=check_nodes_status)
    p.start()
    processes.append(p)

    for p in processes:
        p.join()
