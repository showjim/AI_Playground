from openai import OpenAI

# client = OpenAI(api_key="sk-xxxxxxxxxxxxxxxxxxxx", base_url="https://api.deepseek.com")
client = OpenAI(api_key="sk-xxxxxxxxxxxxxxxxxxxxxxx", base_url="https://api.siliconflow.cn/v1")

# Round 1
while True:
    print("USER Input: ")
    prompt = input()
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model="DeepSeek-R1-pvxei", #"deepseek-ai/DeepSeek-R1", #"deepseek-reasoner",
        messages=messages,
        stream=True
    )

    reasoning_content = ""
    content = ""

    for chunk in response:
        if "reasoning_content" in chunk.choices[0].delta.model_extra:
            if chunk.choices[0].delta.reasoning_content:
                tmp = chunk.choices[0].delta.reasoning_content
                reasoning_content += tmp
                print(f"{tmp}", end='', flush=True)
        if chunk.choices[0].delta.content:
            tmp = chunk.choices[0].delta.content
            content += tmp
            print(f"{tmp}", end='', flush=True)

    messages.append({"role": "assistant", "content": content})
    messages.append({'role': 'user', 'content': prompt})
