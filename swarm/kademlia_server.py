import asyncio
from kademlia.network import Server

async def run_node(port):
    node = Server()
    await node.listen(port)
    try:
        await asyncio.sleep(3600)
    finally:
        node.stop()

if __name__ == "__main__":
    port = 8468
    asyncio.run(run_node(port))
