import streamlit as st
import os, time, json
from typing import List
import azure.cognitiveservices.speech as speechsdk
from src.ClsChatBot import ChatRobot, ChatRobotGemini
import openai

# __version__ = "Beta V0.0.2"
env_path = os.path.abspath(".")
# Azure OpenAI/TTS/STT initial
chatbot = ChatRobot()
chatbot.setup_env()
client = chatbot.initial_llm()
tools = chatbot.initial_tools()
# Gemini initial
chatbot_gemini = ChatRobotGemini()
chatbot_gemini.setup_env()
client_gemini = chatbot_gemini.initial_llm()

st.set_page_config(page_title="GeminiChat - Chatbot With Google AI APIs")


def set_reload_mode():
    st.session_state["GeminiChatReloadMode"] = True


def set_reload_flag():
    # st.write("New document need upload")
    st.session_state["GeminiChatReloadFlag"] = True


# def control_msg_hsitory_szie(msglist: List, max_cnt=10, delcnt=1):
#     while len(msglist) > max_cnt:
#         for i in range(delcnt):
#             msglist.pop(1)
#     return msglist


def main():
    index = 0
    st.title("♊️Gemini Chat Web-UI App")
    # Sidebar contents
    if "GeminiChatReloadMode" not in st.session_state:
        st.session_state["GeminiChatReloadMode"] = True
    if "GeminiChatReloadFlag" not in st.session_state:
        st.session_state["GeminiChatReloadFlag"] = True
    if "GeminiChatAvatarImg" not in st.session_state:
        st.session_state["GeminiChatAvatarImg"] = None
    # Initialize chat history
    if "GeminiChatMessages" not in st.session_state:
        st.session_state["GeminiChatMessages"] = []
    # chain = chatbot.initial_llm()
    if "GeminiChatChain" not in st.session_state:
        # client = chatbot.initial_llm()
        st.session_state["GeminiChatChain"] = client_gemini

    with st.sidebar:
        st.sidebar.expander("Settings")
        st.sidebar.subheader("Parameter for Chatbot")
        aa_chat_mode = st.sidebar.selectbox(label="`0. Chat Mode`",
                                            options=["CasualChat", "Translate", "西瓜一家-小南瓜", "西瓜一家-小东瓜",
                                                     "西瓜一家-Ana"],
                                            index=0,
                                            on_change=set_reload_mode)
        aa_llm_model = st.sidebar.selectbox(label="`1. LLM Model`",
                                            options=["gemini-pro"],
                                            index=0,
                                            on_change=set_reload_flag)
        aa_temperature = st.sidebar.selectbox(label="`2. Temperature (0~1)`",
                                              options=["0", "0.2", "0.4", "0.6", "0.8", "1.0"],
                                              index=1,
                                              on_change=set_reload_flag)
        if "16k" in aa_llm_model:
            aa_max_resp_max_val = 16 * 1024
        else:
            aa_max_resp_max_val = 4096
        aa_max_resp = st.sidebar.slider(label="`3. Max response`",
                                        min_value=256,
                                        max_value=aa_max_resp_max_val,
                                        value=512,
                                        on_change=set_reload_flag)
        aa_context_msg = st.sidebar.slider(label="`4. Context message`",
                                           min_value=5,
                                           max_value=50,
                                           value=20,
                                           on_change=set_reload_flag)

        if st.session_state["GeminiChatReloadMode"] == True:
            system_prompt = chatbot_gemini.select_chat_mode(aa_chat_mode)
            st.session_state["GeminiChatReloadMode"] = False
            # set the tool choice in function call
            if aa_chat_mode == "Translate":
                st.session_state["tool_choice"] = "none"
            else:
                st.session_state["tool_choice"] = "auto"
            # initial the avatar and greeting
            if aa_chat_mode == "西瓜一家-小南瓜":
                st.session_state["GeminiChatAvatarImg"] = "./img/Sunny.png"
                initial_msg = "我是小南瓜，很高兴见到你！"
            else:
                st.session_state["GeminiChatAvatarImg"] = None
                initial_msg = "I'm GeminiChatBot, How may I help you?"
            st.session_state["GeminiChatMessages"] = [
                {"role": "user", "parts": [system_prompt]},
                {"role": "model", "parts": [initial_msg]}
            ]

        if st.session_state["GeminiChatReloadFlag"] == True:
            if "GeminiChatSetting" not in st.session_state:
                st.session_state["GeminiChatSetting"] = {}
            st.session_state["GeminiChatSetting"] = {"model": aa_llm_model, "max_tokens": aa_max_resp,
                                                     "temperature": float(aa_temperature), "context_msg": aa_context_msg}
            st.session_state["GeminiChatReloadFlag"] = False

        # Text2Speech
        aa_voice_name = st.sidebar.selectbox(label="`5. Voice Name`",
                                             options=["None", "小南瓜", "小东瓜", "Ana"],
                                             index=0)
        chatbot.speech_config.speech_recognition_language = "zh-CN"  # "zh-CN" #"en-US"
        if aa_voice_name == "小南瓜":
            aa_voice_name = "zh-CN-XiaoyiNeural"
        elif aa_voice_name == "小东瓜":
            aa_voice_name = "zh-CN-YunxiaNeural"
        elif aa_voice_name == "Ana":
            aa_voice_name = "en-US-AnaNeural"
            chatbot.speech_config.speech_recognition_language = "en-US"  # "zh-CN" #"en-US"

        # Speech2Text
        aa_audio_mode = st.sidebar.selectbox(label="`6. Audio Input Mode`",
                                             options=["Single", "Continuous"],
                                             index=0)
        speech_txt = ""
        if st.sidebar.button("`Speak`"):
            if aa_audio_mode == "Single":
                speech_txt = chatbot.speech_2_text()
            else:
                speech_txt = chatbot.speech_2_text_continous() #speech_2_text() #speech_2_text_continous() #speech_2_text()

    # Display chat messages from history on app rerun
    for message in st.session_state["GeminiChatMessages"]:
        if message["role"] == "user":
            with st.chat_message(message["role"]):
                st.markdown(message["parts"][0])
        elif message["role"] == "model":
            if message["parts"] is not None:
                with st.chat_message(name=message["role"], avatar=st.session_state["GeminiChatAvatarImg"]):
                    st.markdown(message["parts"][0])
                    st.button(label="Play", key="history" + str(index), on_click=chatbot.text_2_speech,
                              args=(message["parts"][0], aa_voice_name,))
                    index += 1

    # Accept user input
    if (prompt := st.chat_input("Type you input here")) or (prompt := speech_txt):
        # Add user message to chat history
        max_cnt = st.session_state["GeminiChatSetting"]["context_msg"]
        st.session_state["GeminiChatMessages"] = chatbot_gemini.control_msg_history_szie(st.session_state["GeminiChatMessages"], max_cnt, 2)
        if st.session_state["GeminiChatMessages"][-1]["role"] == "user":
            # For Gemini Error "400 Please ensure that multiturn requests ends with a user role or a function response."
            st.session_state["GeminiChatMessages"][-1]["parts"] += [prompt]
        else:
            st.session_state["GeminiChatMessages"].append({"role": "user", "parts": [prompt]})

        print("HUMAN: " + prompt)
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant", avatar=st.session_state["GeminiChatAvatarImg"]):
            message_placeholder = st.empty()
            btn_placeholder = st.empty()
            full_response = ""
            with st.spinner("preparing answer"):  # st.session_state["GeminiChatChain"]
                try:
                    generation_config = {
                        "temperature": st.session_state["GeminiChatSetting"]["temperature"],
                        "top_p": 1,
                        "top_k": 1,
                        "max_output_tokens": st.session_state["GeminiChatSetting"]["max_tokens"],
                    }
                    response = st.session_state["GeminiChatChain"].generate_content(
                        contents=st.session_state["GeminiChatMessages"],
                        generation_config=generation_config,
                        stream=True,
                    )
                    for chunk in response:
                        # process normal response and tool_calls response
                        full_response += chunk.text  # ["answer"]  # .choices[0].delta.get("content", "")
                        time.sleep(0.001)
                        message_placeholder.markdown(full_response + "▌")

                except Exception as e:
                    print("Error found: ")
                    print(e)
                    print(st.session_state["GeminiChatMessages"])
                    st.error(e)
                    st.session_state["GeminiChatMessages"].pop(-1)

            message_placeholder.markdown(full_response)

            st.session_state["GeminiChatMessages"].append({"role": "model", "parts": [full_response]})
            print("AI: " + full_response)
            if aa_voice_name != "None":
                chatbot.text_2_speech(full_response, aa_voice_name)
            btn_placeholder.button(label="Play", key="current", on_click=chatbot.text_2_speech,
                                   args=(full_response, aa_voice_name,))


if __name__ == "__main__":
    main()
