import asyncio
import random

async def run():
    async def random_task_sender():
        import aiohttp
        async def post(url, payload):
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.post(
                        url,
                        json=payload,
                        headers={'Content-Type': 'application/json'}
                    ) as resp:
                        return await resp.json()
                except Exception as e:
                    print(f"Error sending random task: {e}")
                    return None

        host, port = ("0.0.0.0", 6050)
        while True:
            task_id = random.randint(0, 10)
            prompt = "Рассккажи историю про пенис."
            generated_text = ""
            gen_ids = []

            payload = {'stage': 0, 'input_data': {'prompt' : prompt}, 'task_id': task_id}
            print(f'#############################################')
            print(f'##########-----------------------############')
            print(f'#############################################')
            print(">>> send:", payload)
            resp = await post(f"http://{host}:{port}/nn_forward", payload)
            if not resp:
                print("Stage0 error")
                continue
            print(f'Resp = {resp}')
            resp = resp['result_for_user']
            nxt = resp.get('next_token_str', "")
            
            gen_ids = resp.get('generated_ids', [])
            if nxt == "":
                print("Received empty token, done.")
                print("Result:", generated_text)
            else:
                generated_text += nxt
                for _ in range(49):
                    payload = {'stage': 0, 'input_data': {'generated_ids': gen_ids}, 'task_id': task_id}
                    print(">>> send:")
                    resp = await post(f"http://{host}:{port}/nn_forward", payload)
                    if not resp:
                        print("Stage0 error")
                        break
                    
                    resp = resp['result_for_user']
                    nxt = resp['next_token_str']
                    gen_ids = resp.get('generated_ids', [])
                    if nxt == "":
                        break
                    generated_text += nxt
                    print(">>> Generated:", generated_text)

                print("Generated text:", generated_text)

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