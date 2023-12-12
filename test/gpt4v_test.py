import os
import requests
import base64
from openai import AzureOpenAI

# Configuration
# GPT4V_KEY = "YOUR_API_KEY"
IMAGE_PATH = r"/Users/jerryzhou/Desktop/generated_00.png" #Untitled.jpg"
encoded_image = base64.b64encode(open(IMAGE_PATH, 'rb').read()).decode('utf-8')
# headers = {
#     "Content-Type": "application/json",
#     "api-key": GPT4V_KEY,
# }

client = AzureOpenAI(
    api_version="2023-12-01-preview",
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT']
)

response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are an AI assistant that helps people find information."
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                        "detail": "low"
                    }
                },
                {
                    "type": "text",
                    "text": "please describe the image with an old Chinese poetry"
                }
            ]
        },
    ],
    max_tokens=512,  # default max tokens is low so set higher
)

# # Payload for the request
# payload = {
#     "messages": [
#         {
#             "role": "system",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": "You are an AI assistant that helps people find information."
#                 }
#             ]
#         },
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": encoded_image
#                     }
#                 },
#                 {
#                     "type": "text",
#                     "text": "Describe the image in detail"
#                 }
#             ]
#         },
#         {
#             "role": "assistant",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": "you are an AI assistant that helps people find information."
#                 }
#             ]
#         }
#     ],
#     "temperature": 0.5,
#     "top_p": 0.95,
#     "max_tokens": 800
# }
#
# GPT4V_ENDPOINT = "https://ai-sh-jerryz-swc.openai.azure.com/openai/deployments/gpt-4v/chat/completions?api-version=2023-07-01-preview"
#
# # Send request
# try:
#     response = requests.post(GPT4V_ENDPOINT, headers=headers, json=payload)
#     response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
# except requests.RequestException as e:
#     raise SystemExit(f"Failed to make the request. Error: {e}")

# Handle the response as needed (e.g., print or process)
resp = response.json()
print(response.choices[0].message.content)
