import asyncio
from kademlia.network import Server
import json


async def client():
    node = Server()
    await node.listen(8469)
    await node.bootstrap([("127.0.0.1", 8468)])
    data = {
            "server_1": 10,
            "server_2": 25,
            "server_3": 5
        }

    json_data = json.dumps(data)
    await node.set("ключ", json_data)
    result = await node.get("ключ")
    print(f"Значение: {result}")

    node.stop()

if __name__ == "__main__":
    asyncio.run(client())


class DistributedHashTableServer:
    def __init__(self, port: int, bootstrap_nodes=None):
        self.port = port
        self.bootstrap_nodes = bootstrap_nodes or []
        self.server = Server()

    async def start(self):
        await self.server.listen(self.port)
        if self.bootstrap_nodes:
            await self.server.bootstrap(self.bootstrap_nodes)

    async def stop(self):
        await self.server.stop()

    async def set(self, key: str, value: dict):
        json_value = json.dumps(value)
        await self.server.set(key, json_value)

    async def get(self, key: str) -> dict:
        result = await self.server.get(key)
        json_dict = json.loads(result) if result else {}
        result = {int(key): value for key, value in json_dict.items()}
        return result

    async def get_all_keys(self):
        return list(self.server.storage.keys())

    async def get_all(self):
        dht_map = {}
        keys = await self.get_all_keys()
        for key in keys:
            if key not in dht_map:
                value = await self.get(key)
                dht_map[key] = value

        return dht_map

    async def run_forever(self):
        while True:
            await asyncio.sleep(3600)
