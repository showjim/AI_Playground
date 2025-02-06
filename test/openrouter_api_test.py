import requests
import json
import asyncio, httpx


# response = requests.post(url, headers=headers, data=json.dumps(payload))
async def call_deepseek(prompt:str):
    # initial
    url = "https://openrouter.ai/api/v1/chat/completions"
    OPENROUTER_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek/deepseek-r1",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "include_reasoning": True,
        "stream": True,
        "Temperature": 0.6,
    }

    reasoning_content = ""
    content = ""
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", url, headers=headers, json=payload) as response:
            async for chunk in response.aiter_lines():
                try:
                    if chunk.startswith("data: "):
                        chunk = chunk[6:]
                    chunk_stripped = chunk.strip()
                    json_chunk = json.loads(chunk_stripped)
                    yield json_chunk
                    if 'choices' in json_chunk and json_chunk['choices']:
                        delta = json_chunk['choices'][0].get('delta', {})
                        if "reasoning" in delta.keys():
                            if delta["reasoning"]:
                                tmp = delta["reasoning"]
                                reasoning_content += tmp
                                print(f"{tmp}", end='', flush=True)
                        if delta["content"]:
                            tmp = delta["content"]
                            content += tmp
                            print(f"{tmp}", end='', flush=True)
                except json.decoder.JSONDecodeError as e:
                    continue
asyncio.run(call_deepseek())

# resp = response.json() #['choices'][0]['message']['reasoning']
# reasoning_content = ""
# content = ""
#
# for chunk in resp:
#     if "reasoning_content" in chunk.choices[0].delta.model_extra:
#         if chunk.choices[0].delta.reasoning_content:
#             tmp = chunk.choices[0].delta.reasoning_content
#             reasoning_content += tmp
#             print(f"{tmp}", end='', flush=True)
#     if chunk.choices[0].delta.content:
#         tmp = chunk.choices[0].delta.content
#         content += tmp
#         print(f"{tmp}", end='', flush=True)
#
# print(response.json()['choices'][0]['message']['reasoning'])
