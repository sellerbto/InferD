import asyncio
import os
import socket
from kademlia_client import DistributedHashTableServer, DHTServer
from node import Node
from partitioned_models import FirstStage, StageInner, LastStage
import yaml

def get_own_ip() -> str:
    node_ip = os.getenv("NODE_IP")
    if node_ip:
        return socket.gethostbyname(node_ip)
    return socket.gethostbyname(socket.gethostname())

def parse_bootstrap_nodes(env_val: str, default_port: int) -> list[tuple[str,int]]:
    nodes = []
    for entry in env_val.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" in entry:
            host, port = entry.split(":", 1)
            nodes.append((host, int(port)))
        else:
            nodes.append((entry, default_port))
    return nodes



async def run_node(node: Node, initial_stage: int):
    try:
        await node.start(initial_stage)
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        await node.stop()
    except Exception as e:
        # logger.error(f"Node failed: {e}")
        raise

async def main():
    with open("inferd.yaml") as f:
        cfg = yaml.safe_load(f)
    node_stages = len(cfg["stages"])
    model_name = cfg["model_name"]
    node_port = 6050
    dht_port = 7050

    rebalance_period = 2
    bootstrap_timeout = 1.0

    node_ip = get_own_ip()
    initial_stage = os.getenv("INITIAL_STAGE")
    if initial_stage is None:
        raise RuntimeError("Переменная окружения INITIAL_STAGE не задана")
    node_name = os.getenv("NODE_NAME")
    if node_name is None:
        raise RuntimeError("Имя ноды NODE_NAME не задано")

    capacity = 0
    bootstrap_nodes = parse_bootstrap_nodes(
        os.getenv("BOOTSTRAP_NODES", ""),
        default_port=dht_port
    )


    dht = DistributedHashTableServer(
        port=dht_port,
        num_stages=node_stages,
        bootstrap_nodes=bootstrap_nodes,
        bootstrap_timeout=bootstrap_timeout
    )
    await dht.start()

    node = Node(
        node_ip,
        node_port,
        node_name,
        model_name,
        initial_stage,
        node_stages,
        capacity,
        dht,
        rebalance_period
    )

    await run_node(node, initial_stage)

if __name__ == "__main__":
    asyncio.run(main())
