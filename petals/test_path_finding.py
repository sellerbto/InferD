import asyncio
import random
import time
import torch

NUM_STAGES = 3
NUM_NODES = 5

async def run():
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

        await asyncio.sleep(25)
        host, port = ("172.28.0.2", 6050)
        while True:
            await asyncio.sleep(20)
            url = f"http://{host}:{port}/nn_forward"
            answer = ''
            r = random.randint(0, 10)
            
            payload = {'stage': 0, 'input_data': 'Привет.', 'task_id': r}
            print(f'#############################################')
            print(f'##########-----------------------############')
            print(f'#############################################')
            start_time = time.time()
            print(f'send task - {payload}, time = {start_time}')
            # print(f'#############################################')
            # print(f'#############################################')
            # print(f'#############################################')
            answ = await post(url, payload)
            # print(f'#############################################')
            # print(f'#############################################')
            # print(f'#############################################')
            end_time = time.time()
            print(f'receive answer = {answ}, time = {end_time}, get answer for {end_time - start_time}')
  
                
            # print(f'#############################################')
            # print(f'#############################################')
            # print(f'#############################################')

    sender = asyncio.create_task(random_task_sender())
    try:
        await asyncio.gather(sender)
    except asyncio.CancelledError:
        print("Shutting down...")

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("Interrupted by user.")