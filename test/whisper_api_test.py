import openai
import time
import os
from src.ClsChatBot import ChatRobot

env_path = os.path.abspath(".")
chatbot = ChatRobot()
# For VS Code key.txt & config.json, in Pycharm use key.txt & config.json
chatbot.setup_env("../key.txt", "../config.json")
#########################
client = chatbot.initial_whisper()

# openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
# openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")  # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
# openai.api_type = "azure"
# openai.api_version = "2023-09-01-preview"
def whisper_STT(audio_test_file="./TalkForAFewSeconds16.wav", audio_language="en"):
    model_name = "whisper"
    result = client.audio.transcriptions.create(
                file=open(audio_test_file, "rb"),
                model=model_name,
                language=audio_language,
                response_format="text",
            )
    return result

if __name__ == '__main__':
    print(whisper_STT("./wikipediaOcelot.wav"))