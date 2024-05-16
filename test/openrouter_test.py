from openai import OpenAI
from os import getenv

# gets API Key from environment variable OPENAI_API_KEY
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key= getenv("OPENROUTER_API_KEY"),
)

completion = client.chat.completions.create(
    model="openchat/openchat-7b:free",
    messages=[
        {"role": "system", "content": "You are an AI assistant."},
        {'role': 'assistant', 'content': "I'm OpenChatBot, How may I help you?"},
        {"role": "user", "content": "Say this is a test",},
    ],
    max_tokens=512,
    temperature=0.2,
    stream=True,
)
# print(completion.choices[0].message.content)
full_response = ""
for chunk in completion:
    # process normal response and tool_calls response
    if len(chunk.choices) > 0:
        deltas = chunk.choices[0].delta
        if deltas.content is not None:
            full_response += deltas.content  # ["answer"]  # .choices[0].delta.get("content", "")
            print(full_response)