import subprocess
from ollama import Client
from time import time, sleep

import requests

PROMPT = """Please translate the following text into French. Follow these guidelines:
1. Maintain the original layout and formatting.
2. Translate all text accurately without omitting any part of the content.
3. Preserve the tone and style of the original text.
4. Do not include any additional comments, notes, or explanations in the output; provide only the translated text.
5. Only translate the text between ``` and ```. Do not output any other text or character.

Here is the text to be translated:

With irony and mischief the young Czech artist Krištof Kintera turns art and life on their heads.
"""


def check_times():
    ollama_model = "aya:35b"
    lms_model = "coherelabs_aya-23-35b"
    _ = Client().chat(model=ollama_model, messages=[{"role": "user", "content": PROMPT}])

    start_time = time()
    for i in range(100):
        Client().chat(model=ollama_model, messages=[{"role": "user", "content": PROMPT}])
    print(f"Ollama Time: {round(time() - start_time, 2)} seconds")

    subprocess.run(["ollama", "stop", ollama_model])
    sleep(2)

    # subprocess.run(["lms", "load", "openai/gpt-oss-20b"])
    subprocess.run(["lms", "load", lms_model])
    _ = requests.post(
        "http://localhost:3000/v1/chat/completions",
        json={
            "model": lms_model,
            "messages": [{"role": "user", "content": PROMPT}],
        },
    )

    start_time = time()
    for i in range(100):
        requests.post(
            "http://localhost:3000/v1/chat/completions",
            json={
                "model": lms_model,
                "messages": [{"role": "user", "content": PROMPT}],
            },
        )
    print(f"LM Studio Time: {round(time() - start_time, 2)} seconds")
    subprocess.run(["lms", "unload", "--all"])


if __name__ == "__main__":
    check_times()

    # llama3.1 (100 requests):
    # Ollama Time: 41.6 seconds
    # LM Studio Time: 36.37 seconds

    # llama3.1 (1000 requests):
    # Ollama Time: 403.64 seconds
    # LM Studio Time: 373.17 seconds

    # gpt-oss (100 requests):
    # Ollama Time: 457.73 seconds
    # LM Studio Time: 179.24 seconds

    # aya:35b (100 requests):
    # Ollama Time: 552.92 seconds
    # LM Studio Time: 535.14 seconds

    # https://huggingface.co/blog/yagilb/lms-hf
