import os, io
from src.ClsChatBot import ChatRobot
import streamlit as st
from streamlit_mic_recorder import mic_recorder,speech_to_text

env_path = os.path.abspath(".")
chatbot = ChatRobot()
# For VS Code key.txt & config.json, in Pycharm use key.txt & config.json
chatbot.setup_env("../key.txt", "../config.json")
#########################
client_stt = chatbot.initial_whisper()

# openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
# openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")  # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
# openai.api_type = "azure"
# openai.api_version = "2023-09-01-preview"
def whisper_STT(audio_test_file="./TalkForAFewSeconds16.wav", audio_language="en"):
    model_name = "whisper-1"
    result = client_stt.audio.transcriptions.create(
                file=audio_test_file, #open(audio_test_file, "rb"),
                model=model_name,
                language=audio_language,
                response_format="verbose_json", #"text",
            )
    return result

def main():
    state = st.session_state

    if 'text_received' not in state:
        state.text_received = []

    st.write("Record your voice, and play the recorded audio:")
    audio = mic_recorder(start_prompt="⏺️", stop_prompt="⏹️", key='recorder')

    if audio:
        st.audio(audio['bytes'])
        audio_BIO = io.BytesIO(audio['bytes'])
        audio_BIO.name = 'audio.mp3'
        st.write(whisper_STT(audio_BIO, "zh"))

if __name__ == '__main__':
    print(whisper_STT(open("./wikipediaOcelot.wav","rb"), "en"))
    # main()