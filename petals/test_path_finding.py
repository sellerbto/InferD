import asyncio
import random
import time
import csv
from config import port_shift
import traceback
from kademlia_client import DistributedHashTableServer, DHTServer
from node import Node
import aiohttp

METRICS_LOG_PATH = "metrics_log.csv"
NUM_STAGES = 3
NUM_NODES = 5

async def collect_and_log(bootstrap_nodes, num_stages, interval):
    await asyncio.sleep(8)
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

    task_bootstrap = [(host, port + 500) for host, port in bootstrap_nodes]
    task_dht = DHTServer(0,
                         bootstrap_nodes=task_bootstrap)
    await task_dht.start()
    start_time = time.time()

    try:
        while True:
            t = time.time() - start_time
            dht_map = await dht.get_all()
            row = [f"{t:.2f}"]
            for stage in range(num_stages):
                peers = dht_map.get(str(stage), {})
                if peers:
                    loads = [p['load'] for p in peers.values()]
                    caps  = [p['cap']  for p in peers.values()]
                    row.append(f"{min(loads):.1f}")
                    row.append(str(sum(caps)))
                else:
                    row.extend(['', '0'])
            for stage in range(num_stages):
                row.append(str(sum((await task_dht.get(stage)).values())))
            all_servers_count = 0
            for stage in range(num_stages):
                peers = dht_map.get(str(stage), {})
                row.append(str(len(peers)))
                all_servers_count += len(peers)


            with open(METRICS_LOG_PATH, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

            await asyncio.sleep(interval)
    finally:
        await dht.stop()


def start_node(node_id, port, bootstrap_nodes, num_stages, rebalance_period):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    ports = [6000 + i for i in range(NUM_NODES)]
    async def main_node():
        dht = DistributedHashTableServer(port, num_stages=num_stages, bootstrap_nodes=bootstrap_nodes)
        await dht.start()

        if node_id == 0:
            for stage in range(num_stages):
                init_dict = {
                    f'127.0.0.1:{ports[n]}': {'load': 0, 'cap': 0}
                    for n in range(1, NUM_NODES)
                }
                await dht.set(str(stage), init_dict)
            print("Node 0: DHT pre‐initialized")

        task_bootstrap = [(host, port + 500) for host, port in bootstrap_nodes]
        task_dht = DHTServer(port=port+500,
                            bootstrap_nodes=task_bootstrap)
        await task_dht.start()

        capacity = 3
        initial_stage = node_id % num_stages
        endpoint_port = port+port_shift
        node = Node(endpoint_port,
                    num_stages,
                    capacity,
                    dht, 
                    task_dht, 
                    rebalance_period)
        await node.start(initial_stage)
        print(f"🟢 Node {node.node_info.id}: DHT port={node.node_info.port}, HTTP port={endpoint_port}, stage={initial_stage}, cap={capacity}")
        await asyncio.Event().wait()
    try:
        loop.run_until_complete(main_node())
    except Exception:
        traceback.print_exc()
    finally:
        # try:
        #     loop.run_until_complete(dht.stop())
        # except Exception:
        #     pass
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

    collector = asyncio.create_task(
        collect_and_log(
            bootstrap_nodes=[("127.0.0.1", ports[0])],
            num_stages=NUM_STAGES,
            interval=rebalance_period
        )
    )

    async def random_task_sender():
        import aiohttp
        
        async def post(url, payload):
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.post(url, json=payload) as resp:
                        data = await resp.json()
                        return data
                except Exception as e:
                    print(f"Error sending random task: {e}")
                    return None

        await asyncio.sleep(15)
        host, port = ("127.0.0.1", ports[0])
        url = f"http://{host}:{port+port_shift}/nn_forward"
        r = random.randint(0, 10)
        payload = {'stage': 0, 'input_data': 0, 'task_id': r}
        print(f'#############################################')
        print(f'#############################################')
        print(f'#############################################')
        start_time = time.time()
        print(f'send task - {payload}, time = {start_time}')
        print(f'#############################################')
        print(f'#############################################')
        print(f'#############################################')
        answ = await post(url, payload)

        print(f'#############################################')
        print(f'#############################################')
        print(f'#############################################')
        end_time = time.time()
        print(f'receive answer = {answ}, time = {end_time}, get answer for {end_time - start_time}')
        print(f'#############################################')
        print(f'#############################################')
        print(f'#############################################')

    sender = asyncio.create_task(random_task_sender())
    try:
        await asyncio.gather(collector, sender)
    except asyncio.CancelledError:
        print("Shutting down...")

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("Interrupted by user.")