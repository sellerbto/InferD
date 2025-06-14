import asyncio
import aiohttp
import sys
import uuid
import time
from transformers import AutoTokenizer
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.shortcuts import print_formatted_text
from colorama import init, Fore, Style as ColorStyle

# Initialize colorama
init(autoreset=True)

# Load Qwen-2-0.5B tokenizer for chat templating
TOKENIZER_NAME = "Qwen/Qwen2-0.5B"
print(f"Loading tokenizer '{TOKENIZER_NAME}'...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
print("Tokenizer loaded.\n")

# HTTP endpoint
HOST = "0.0.0.0"
PORT = 6050
URL = f"http://{HOST}:{PORT}/nn_forward"

# Prompt toolkit style
def get_style():
    return Style.from_dict({
        'prompt': '#00aa00 bold',
        'ai': '#0055ff italic',
        'error': '#ff0000 bold'
    })

async def send_request(session, payload):
    try:
        async with session.post(URL, json=payload, headers={"Content-Type": "application/json"}) as resp:
            return await resp.json()
    except Exception as e:
        print_formatted_text(HTML(f"<error>[Error] {e}</error>"))
        return None

async def chat():
    session = PromptSession()
    style = get_style()
    print_formatted_text(HTML("<ai>🤖 Welcome! Ask your question.</ai>"))

    async with aiohttp.ClientSession() as http_session:
        while True:
            try:
                with patch_stdout():
                    user_msg = await session.prompt_async(HTML('<prompt>You:</prompt> '), style=style)
            except (EOFError, KeyboardInterrupt):
                print_formatted_text(HTML("<ai>👋 Goodbye!</ai>"))
                return

            prompt_text = user_msg.strip()
            if prompt_text.lower() in ("exit", "quit"):
                print_formatted_text(HTML("<ai>👋 Goodbye!</ai>"))
                return

            # Build messages and apply chat template
            messages = [{"role": "user", "content": prompt_text}]
            template = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )

            task_id = str(uuid.uuid4())
            payload = {'stage': 0, 'input_data': {'prompt': template}, 'task_id': task_id}
            print_formatted_text(HTML("<status>[Sending request...]</status>"))
            response = await send_request(http_session, payload)

            if not response or 'result_for_user' not in response:
                print_formatted_text(HTML("<error>[Error] No response.</error>"))
                continue

            result = response['result_for_user']
            gen_ids = result.get('generated_ids', [])
            next_token = result.get('next_token_str', '')

            print_formatted_text(HTML('<ai>AI:</ai> '), end='')

            # Stream with inline token count and timer (non-disruptive)
            start_time = time.time()
            token_count = 0
            try:
                while next_token:
                    # Print the token
                    print(next_token, end='', flush=True)
                    token_count += 1
                    elapsed = time.time() - start_time
                    # Print inline count/time in dim color
                    print(ColorStyle.DIM + f"[{token_count} tokens | {elapsed:.2f}s]" + ColorStyle.RESET_ALL, end='', flush=True)

                    # Request next token
                    payload = {'stage': 0, 'input_data': {'generated_ids': gen_ids}, 'task_id': task_id}
                    response = await send_request(http_session, payload)
                    if not response or 'result_for_user' not in response:
                        print_formatted_text(HTML("\n<error>[Stream error]</error>"))
                        break
                    data = response['result_for_user']
                    next_token = data.get('next_token_str', '')
                    gen_ids = data.get('generated_ids', [])
            except KeyboardInterrupt:
                print()  # newline
                print_formatted_text(HTML("<error>⏹️ Interrupted.</error>"))

            # Ensure newline after answer
            print()  

if __name__ == "__main__":
    try:
        asyncio.run(chat())
    except KeyboardInterrupt:
        print(Fore.YELLOW + "\n[Exited by user]" + ColorStyle.RESET_ALL)
        sys.exit(0)