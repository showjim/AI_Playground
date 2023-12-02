import streamlit as st
# from streamlit_chat import message
from src.chat import CasualChatBot, StreamHandler
import os, time
from langchain.callbacks import StreamlitCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import azure.cognitiveservices.speech as speechsdk


# __version__ = "Beta V0.0.2"

env_path = os.path.abspath('.')
casual_chat_bot = CasualChatBot(env_path)
# casual_chat_bot.setup_langchain() #change to st, then we can use memory function
speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SPEECH_KEY'),
                                       region=os.environ.get('SPEECH_REGION'))
audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

st.set_page_config(page_title="CasualChat -  Personal Chatbot based on Streamlit")

# Response output
## Function for taking user prompt as input followed by producing AI generated responses
def generate_response(prompt):
    # casual_chat_bot.chat_langchain(prompt) #chatbot.chat(prompt)
    # change to here, call predict from st.session_state["chain"] to use memory function from langchain
    response = st.session_state["chain"].predict(human_input=prompt)
    # print(response)
    return response

def set_reload_flag():
    # st.write("New document need upload")
    st.session_state["casualchatreloadflag"] = True

def text_2_speech(text:str):
    # The language of the voice that speaks.
    speech_config.speech_synthesis_voice_name = "zh-CN-XiaoyiNeural"  # "zh-CN-YunxiaNeural"
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()

    if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesized for text [{}]".format(text))
    elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_synthesis_result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            if cancellation_details.error_details:
                print("Error details: {}".format(cancellation_details.error_details))
                print("Did you set the speech resource key and region values?")

def main():
    st.title('💬 Casual Chat Web-UI App')
    # Sidebar contents
    if "casualchatreloadflag" not in st.session_state:
        st.session_state["casualchatreloadflag"] = None
    with st.sidebar:
        st.sidebar.expander("Settings")
        st.sidebar.subheader("Parameter for Chatbot")
        aa_llm_model = st.sidebar.selectbox(label="1. LLM Model",
                                              options=["gpt-35-turbo", "gpt-35-turbo-16k", "gpt-4", "gpt-4-turbo"],
                                              index=0,
                                              on_change=set_reload_flag)
        aa_temperature = st.sidebar.selectbox(label="2. Temperature (0~1)",
                                              options=["0", "0.2", "0.4", "0.6", "0.8", "1.0"],
                                              index=1,
                                              on_change=set_reload_flag)
        if "16k" in aa_llm_model:
            aa_max_resp_max_val = 16*1024
        else:
            aa_max_resp_max_val = 4096
        aa_max_resp = st.sidebar.slider(label="3. Max response",
                                        min_value=256,
                                        max_value=aa_max_resp_max_val,
                                        value=2048,
                                        on_change=set_reload_flag)
        if "chain" not in st.session_state or st.session_state["casualchatreloadflag"] == True:
            chain = casual_chat_bot.initial_llm("CasualChat", aa_llm_model, aa_max_resp, float(aa_temperature))
            st.session_state["chain"] = chain
            st.session_state["casualchatreloadflag"] = False

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", "content": "I'm CasualChat, How may I help you?"}]

    # Display chat messages from history on app rerun
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Type you input here"):
        # Add user message to chat history
        st.session_state["messages"].append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st_callback = StreamHandler(st.container()) #StreamingStdOutCallbackHandler StreamlitCallbackHandler(st.container()) StreamHandler(st.container())
            message_placeholder = st.empty()
            full_response = ""
            with st.spinner('preparing answer'):
                # full_response = st.session_state["chain"].predict(human_input=prompt, callbacks=[st_callback])
                for response in st.session_state["chain"].predict(human_input=prompt, callbacks=[st_callback]):
                    full_response += response #.choices[0].delta.get("content", "")
                    time.sleep(0.001)
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            btn = st.button(label="Play", on_click=text_2_speech, args=(full_response,))

        st.session_state['messages'].append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()