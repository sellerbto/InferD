# test.py

import asyncio
import random
import time
import os
import csv
from config import port_shift
from kademlia_client import DistributedHashTableServer
from node import Node
from test_utils import task_distribution

METRICS_LOG_PATH = "metrics_log.csv"
NUM_STAGES = 3
NUM_NODES = 5

task_distribution = {stage: 0  for stage in range(NUM_STAGES)}

# test.py (updated collect_and_log to log per-stage server counts)

async def collect_and_log(bootstrap_nodes, num_stages, interval):
    from test_utils import task_distribution
    await asyncio.sleep(5)

    # Initialize CSV: добавляем колонки для task_distribution и server_count
    with open(METRICS_LOG_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        headers = ['time_s']
        for stage in range(num_stages):
            headers.append(f'stage{stage}_min_load')
        for stage in range(num_stages):
            headers.append(f'stage{stage}_total_cap')
        for stage in range(num_stages):
            headers.append(f'stage{stage}_tasks_running')
        for stage in range(num_stages):
            headers.append(f'stage{stage}_servers')
        writer.writerow(headers)

    dht = DistributedHashTableServer(port=0, num_stages=num_stages, bootstrap_nodes=bootstrap_nodes)
    await dht.start()
    start_time = time.time()

    try:
        while True:
            t = time.time() - start_time
            dht_map = await dht.get_all()
            row = [f"{t:.2f}"]
            for stage in range(num_stages):
                peers = dht_map.get(str(stage), {})
                # print(peers)
                if peers:
                    loads = [p['load'] for p in peers.values()]
                    caps  = [p['cap']  for p in peers.values()]
                    row.append(f"{min(loads):.1f}")
                    row.append(str(sum(caps)))
                else:
                    row.extend(['', '0'])
            # tasks_running
            for stage in range(num_stages):
                row.append(str(task_distribution.get(stage, 0)))
            # servers count
            all_servers_count = 0
            for stage in range(num_stages):
                peers = dht_map.get(str(stage), {})
                row.append(str(len(peers)))
                all_servers_count += len(peers)
            
            assert all_servers_count == NUM_NODES, f'{all_servers_count} != {NUM_NODES}'

            with open(METRICS_LOG_PATH, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

            await asyncio.sleep(interval)
    finally:
        await dht.stop()


def start_node(node_id, port, bootstrap_nodes, num_stages, rebalance_period):
    import asyncio, traceback, random
    # Create and run loop in this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    async def main_node():
        # Start DHT on given port
        dht = DistributedHashTableServer(port, num_stages=num_stages, bootstrap_nodes=bootstrap_nodes)
        await dht.start()
        # Random capacity and initial stage
        capacity = 3
        initial_stage = node_id % num_stages
        endpoint_port = port+port_shift
        node = Node(endpoint_port, num_stages, capacity, dht, rebalance_period)
        await node.start(initial_stage)
        print(f"🟢 Node {node.node_id}: DHT port={node.port}, HTTP port={endpoint_port}, stage={initial_stage}, cap={capacity}")
        # Keep alive
        await asyncio.Event().wait()
    try:
        loop.run_until_complete(main_node())
    except Exception:
        traceback.print_exc()
    finally:
        try:
            loop.run_until_complete(dht.stop())
        except Exception:
            pass
        loop.close()

async def run():
    rebalance_period = 2.0

    ports = [6000 + i for i in range(NUM_NODES)]
    bootstrap_for = {0: []}
    for nid in range(1, NUM_NODES):
        bootstrap_for[nid] = [("127.0.0.1", ports[0])]

    import threading
    threads = []
    for node_id in range(NUM_NODES):
        t = threading.Thread(
            target=start_node,
            args=(node_id, ports[node_id], bootstrap_for[node_id], NUM_STAGES, rebalance_period),
            daemon=True
        )
        t.start()
        threads.append(t)
        await asyncio.sleep(0.5)

    print(f"Started {NUM_NODES} nodes with {NUM_STAGES} stages.")

    # Запуск сборщика метрик
    collector = asyncio.create_task(
        collect_and_log(
            bootstrap_nodes=[("127.0.0.1", ports[0])],
            num_stages=NUM_STAGES,
            interval=rebalance_period
        )
    )

    async def random_task_sender():
        import aiohttp
        async with aiohttp.ClientSession() as session:
            stage = 0
            while True:
                # stage = random.randint(0, NUM_STAGES - 1)
                node_idx = random.randint(0, NUM_NODES - 1)
                host, port = ("127.0.0.1", ports[node_idx])
                url = f"http://{host}:{port+port_shift}/nn_forward"
                payload = {'stage': stage % NUM_STAGES, 'input_data': {}}

                try:
                    async with session.post(url, json=payload) as resp:
                        data = await resp.json()
                        print(f"Random task: stage {stage} -> node {data['node_id']}")
                        task_distribution[stage % NUM_STAGES] = task_distribution.get(stage % NUM_STAGES, 0) + 1
                except Exception as e:
                    print(f"Error sending random task: {e}")

                await asyncio.sleep(5)
                # stage += 1

    sender = asyncio.create_task(random_task_sender())

    # Ждём только по collector и sender (ноды в потоках)
    try:
        await asyncio.gather(collector, sender)
    except asyncio.CancelledError:
        print("Shutting down...")

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("Interrupted by user.")