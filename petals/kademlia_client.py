# kademlia_client.py
import asyncio
import json
import logging
from kademlia.network import Server

logger = logging.getLogger("Kademlia")

class DistributedHashTableServer:
    def __init__(
        self,
        port: int,
        num_stages: int,
        bootstrap_nodes=None,
        bootstrap_timeout: float = 1.0  # Добавлен параметр таймаута
    ):
        self.port = port
        self.bootstrap_nodes = bootstrap_nodes or []
        self.num_stages = num_stages
        self.bootstrap_timeout = bootstrap_timeout
        self.server = Server()

    async def start(self):
        await self.server.listen(self.port)
        for attempt in range(5):                             # до 5 попыток
            try:
                await asyncio.wait_for(
                    self.server.bootstrap(self.bootstrap_nodes),
                    timeout=self.bootstrap_timeout
                )
                logger.info(f"Bootstrapped on attempt {attempt+1}")
                break
            except asyncio.TimeoutError:
                logger.warning(f"Bootstrap attempt {attempt+1} timed out")
                await asyncio.sleep(self.bootstrap_timeout)
        else:
            logger.error("All bootstrap attempts failed")


    async def stop(self):
        await self.server.stop()

    async def set(self, key: str, value: dict, timeout: float = 5.0):
        json_value = json.dumps(value)
        try:
            await asyncio.wait_for(
                self.server.set(key, json_value),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Set operation timed out for key: {key}")
        except Exception as e:
            logger.error(f"Set error: {e}")

    async def get(self, key: str, timeout: float = 5.0) -> dict:
        if not self.server.bootstrappable_neighbors():
            return {}
        try:
            result = await asyncio.wait_for(
                self.server.get(key),
                timeout=timeout
            )
            return json.loads(result) if result else {}
        except asyncio.TimeoutError:
            logger.warning(f"Get operation timed out for key: {key}")
            return {}
        except Exception as e:
            logger.error(f"Get error: {e}")
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

class DHTServer:
    def __init__(
        self,
        port: int,
        bootstrap_nodes=None,
        bootstrap_timeout: float = 1.0  # Добавлен параметр таймаута
    ):
        self.port = port
        self.bootstrap_nodes = bootstrap_nodes or []
        self.bootstrap_timeout = bootstrap_timeout
        self.keys = set()
        self.server = Server()

    async def start(self):
        await self.server.listen(self.port)
        if self.bootstrap_nodes:
            await asyncio.wait_for(
                self.server.bootstrap(self.bootstrap_nodes),
                timeout=self.bootstrap_timeout  # Используем параметр
            )

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

    async def get(self, key: str, timeout: int = 1) -> dict:
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
