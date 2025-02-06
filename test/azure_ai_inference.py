from azure.ai.inference import ChatCompletionsClient, EmbeddingsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage

client = ChatCompletionsClient(
    endpoint='https://DeepSeek-R1-pvxei.eastus.models.ai.azure.com',
    credential=AzureKeyCredential("epJ1ACLL5W5IM0EeawFWpK5mApX1qEDW"),
    api_version="2024-06-01",  # Azure OpenAI api-version. See https://aka.ms/azsdk/azure-ai-inference/azure-openai-api-versions
)
model_info = client.get_model_info()
print("Model name:", model_info.model_name)
print("Model type:", model_info.model_type)
print("Model provider name:", model_info.model_provider_name)

response = client.complete(
        messages=[
            SystemMessage("You are a helpful assistant."),
            UserMessage("How many feet are in a mile?"),
        ]
    )

print(response.choices[0].message.content)