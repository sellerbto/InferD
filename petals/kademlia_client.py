# kademlia_client.py

import asyncio
from kademlia.network import Server
import json
from config import port_shift

class DistributedHashTableServer:
    def __init__(self, port: int, num_stages: int, bootstrap_nodes=None):
        self.port = port
        self.bootstrap_nodes = bootstrap_nodes or []
        self.num_stages = num_stages
        self.server = Server()

    async def start(self):
        await self.server.listen(self.port)
        if self.bootstrap_nodes:
            try:
                await asyncio.wait_for(
                    self.server.bootstrap(self.bootstrap_nodes),
                    timeout=2.0
                )
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                pass
        for stage in range(self.num_stages):
            await self.set(str(stage), {})

    async def stop(self):
        await self.server.stop()

    async def set(self, key: str, value: dict):
        json_value = json.dumps(value)
        try:
            await self.server.set(key, json_value)
        except Exception as e:
            pass

    async def get(self, key: str, timeout: int = 5) -> dict:
        if not self.server.bootstrappable_neighbors():
            return {}
        try:
            result = await asyncio.wait_for(self.server.get(key), timeout=timeout)
            json_dict = json.loads(result) if result else {}
            return json_dict
        except asyncio.TimeoutError:
            return {}
        except Exception as e:
            return {}

    async def get_all_keys(self):
        keys = [str(i) for i in range(self.num_stages)]

        return keys

    async def get_all(self):
        if not self.server.bootstrappable_neighbors():
            return {}
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


class DHTServer:
    def __init__(self, port: int, bootstrap_nodes=None):
        self.port = port
        self.bootstrap_nodes = bootstrap_nodes or []
        self.keys = set()
        self.server = Server()

    async def start(self):
        await self.server.listen(self.port)
        if self.bootstrap_nodes:
            try:
                await asyncio.wait_for(
                    self.server.bootstrap(self.bootstrap_nodes),
                    timeout=2.0
                )
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                pass


    async def stop(self):
        await self.server.stop()

    async def set(self, key: str, value: dict):
        # if key is not str:
        #     raise Exception
        self.keys.add(key)
        json_value = json.dumps(value)
        try:
            await self.server.set(key, json_value)
        except Exception as e:
            pass

    async def get(self, key: str, timeout: int = 5) -> dict:
        if not self.server.bootstrappable_neighbors():
            return {}
        try:
            result = await asyncio.wait_for(self.server.get(key), timeout=timeout)
            json_dict = json.loads(result) if result else {}
            return json_dict
        except asyncio.TimeoutError:
            return {}
        except Exception as e:
            return {}

    async def get_all_keys(self):
        return list(self.keys)

    async def get_all(self):
        if not self.server.bootstrappable_neighbors():
            return {}
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