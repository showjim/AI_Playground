from openai import OpenAI
from os import getenv

# gets API Key from environment variable OPENAI_API_KEY
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key= getenv("OPENROUTER_API_KEY"),
)

completion = client.chat.completions.create(
    model="google/gemini-flash-1.5",
    messages=[
        {
          "role": "user",
          "content": "Say this is a test",
        },
    ],
)
print(completion.choices[0].message.content)