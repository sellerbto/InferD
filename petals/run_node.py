import asyncio
import os
import socket
import logging
from kademlia_client import DistributedHashTableServer, DHTServer
from node import Node

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Main")

def get_own_ip() -> str:
    node_ip = os.getenv("NODE_IP")
    if node_ip:
        try:
            return socket.gethostbyname(node_ip)
        except socket.gaierror:
            logger.warning("Failed to resolve container name, using default")
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
            # тут default_port = DHT-порт (7050)
            nodes.append((entry, default_port))
    return nodes



async def run_node(node: Node, initial_stage: int):
    try:
        await node.start(initial_stage)
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        await node.stop()
    except Exception as e:
        logger.error(f"Node failed: {e}")
        raise

async def main():
    node_stages = 3
    node_port = 6050
    dht_port = 7050
    task_dht_port = 7051
    capacity = 100
    rebalance_period = 10
    bootstrap_timeout = 4.0  # Увеличенный таймаут

    node_ip = get_own_ip()
    initial_stage = int(os.getenv("INITIAL_STAGE", 0))
    bootstrap_nodes = parse_bootstrap_nodes(
        os.getenv("BOOTSTRAP_NODES", ""),
        default_port=dht_port     # dht_port = 7050
    )
    task_bootstrap_nodes = parse_bootstrap_nodes(
        os.getenv("TASK_BOOTSTRAP_NODES", ""),
        default_port=task_dht_port  # task_dht_port = 7051
    )

    logger.info(f"Starting node. IP: {node_ip}, Port: {node_port}, Stage: {initial_stage}")

    try:
        dht = DistributedHashTableServer(
            port=dht_port,
            num_stages=node_stages,
            bootstrap_nodes=bootstrap_nodes,
            bootstrap_timeout=bootstrap_timeout
        )
        await dht.start()

        task_dht = DHTServer(
            port=task_dht_port,
            bootstrap_nodes=task_bootstrap_nodes,
            bootstrap_timeout=bootstrap_timeout
        )
        await task_dht.start()

        node = Node(
            node_ip,
            node_port,
            node_stages,
            capacity,
            dht,
            task_dht,
            rebalance_period
        )

        await run_node(node, initial_stage)

    except Exception as e:
        logger.error(f"Main failed: {e}")
        await dht.stop()
        await task_dht.stop()
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise