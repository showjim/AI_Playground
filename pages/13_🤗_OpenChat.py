import hmac
from typing import List

import streamlit as st
import os, time, json, io, httpx, asyncio
from src.ClsChatBot import ChatRobotOpenRouter
# from HomePage import __version__


env_path = os.path.abspath("..")

chatbot = ChatRobotOpenRouter()
chatbot.setup_env("./setting/key.txt", "./setting/config.json")
client = chatbot.initial_llm()




def set_reload_mode():
    st.session_state["OpenChatReloadMode"] = True


def set_reload_flag():
    # st.write("New document need upload")
    st.session_state["OpenChatReloadFlag"] = True

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False) or st.secrets["password"] == "":
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False

def display_message(messages):
    for message in messages:
        if message["role"] == "user":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        elif message["role"] == "assistant":
            if message["content"] is not None:
                with st.chat_message(name=message["role"], avatar=st.session_state["OpenAvatarImg"]):
                    if message["content"]["thinking"] != "":
                        with st.expander("See thinking"):
                            st.markdown(message["content"]["thinking"])
                    st.markdown(message["content"]["answers"])

async def call_openrouter(model:str, messages: List, temperature:float, max_tokens:int, enable_reasoning:bool):
    # initial
    url = "https://openrouter.ai/api/v1/chat/completions"
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "include_reasoning": enable_reasoning,
        "stream": True,
        "Temperature": temperature,
        "max_tokens": max_tokens,
    }

    async with httpx.AsyncClient() as client:
        async with client.stream("POST", url, headers=headers, json=payload, timeout=60) as response:
            async for chunk in response.aiter_lines():
                try:
                    if chunk.startswith("data: "):
                        chunk = chunk[6:]
                    chunk_stripped = chunk.strip()
                    json_chunk = json.loads(chunk_stripped)
                    yield json_chunk
                    # if 'choices' in json_chunk and json_chunk['choices']:
                    #     delta = json_chunk['choices'][0].get('delta', {})
                    #     if "reasoning" in delta.keys():
                    #         if delta["reasoning"]:
                    #             tmp = delta["reasoning"]
                    #             reasoning_content += tmp
                    #             print(f"{tmp}", end='', flush=True)
                    #     if delta["content"]:
                    #         tmp = delta["content"]
                    #         content += tmp
                    #         print(f"{tmp}", end='', flush=True)
                except json.decoder.JSONDecodeError as e:
                    continue

async def main():
    index = 0
    st.title("🤗Open Chat Web-UI App ")
    st.caption('Powered by Streamlit, written by Chao Zhou')
    st.subheader("", divider='rainbow')

    if not check_password():
        st.stop()  # Do not continue if check_password is not True.

    # Sidebar contents
    if "OpenChatReloadMode" not in st.session_state:
        st.session_state["OpenChatReloadMode"] = True
    if "OpenChatReloadFlag" not in st.session_state:
        st.session_state["OpenChatReloadFlag"] = True
    if "OpenAvatarImg" not in st.session_state:
        st.session_state["OpenAvatarImg"] = None
    # Initialize chat history
    if "OpenChatMessages" not in st.session_state:
        st.session_state["OpenChatMessages"] = []
    if "OpenChatMessagesDispay" not in st.session_state:
        # this is a shadow of "OpenChatMessages" to keep image URL from Dalle3
        st.session_state["OpenChatMessagesDispay"] = []

    with st.sidebar:
        st.header("Other Tools")
        st.page_link("http://taishanstone:8501", label="Check INFO Tool", icon="1️⃣")
        st.page_link("http://taishanstone:8502", label="Pattern Auto Edit Tool", icon="2️⃣")
        st.page_link("http://taishanstone:8503", label="Shmoo Detect Tool", icon="3️⃣")
        st.header("Help")
        if st.button("About"):
            st.info(
                "Thank you for using!\nCreated by Chao Zhou.\nAny suggestions please mail zhouchao486@gmail.com]")

        with st.expander("Settings"):
            st.subheader("Parameter for Chatbot")
            aa_chat_mode = st.selectbox(label="`0. Chat Mode`",
                                                options=["CasualChat", "Meta-prompt", "Translate", "Thinking Protocol"],
                                                index=0,
                                                on_change=set_reload_mode)
            aa_llm_model = st.selectbox(label="`1. LLM Model`",
                                        options=["openchat/openchat-7b:free",
                                                 "anthropic/claude-3-5-haiku",
                                                 "anthropic/claude-3.5-sonnet",
                                                 "qwen/qwen-2.5-coder-32b-instruct",
                                                 "deepseek/deepseek-chat",
                                                 "deepseek/deepseek-r1",
                                                 "deepseek/deepseek-r1:free",
                                                 "openai/gpt-4o-mini",
                                                 "openai/gpt-4o"
                                                 ],
                                        index=0,
                                        on_change=set_reload_flag)
            # initial the avatar and greeting
            if "openchat" in aa_llm_model:
                st.session_state["OpenAvatarImg"] = "./img/logo/openchat-logo.png"
            elif "meta-llama" in aa_llm_model:
                st.session_state["OpenAvatarImg"] = "./img/logo/meta-logo.png"
            elif "google" in aa_llm_model:
                st.session_state["OpenAvatarImg"] = "./img/logo/google-logo.png"
            elif "mistralai" in aa_llm_model:
                st.session_state["OpenAvatarImg"] = "./img/logo/mistral-logo.png"
            elif "openai" in aa_llm_model:
                st.session_state["OpenAvatarImg"] = "./img/logo/openai-logo.png"
            elif "deepseek" in aa_llm_model:
                st.session_state["OpenAvatarImg"] = "./img/logo/deepseek-logo.png"
            elif "qwen" in aa_llm_model:
                st.session_state["OpenAvatarImg"] = "./img/logo/qwen-logo.png"
            elif "claude" in aa_llm_model:
                st.session_state["OpenAvatarImg"] = "./img/logo/Claude-logo.png"
            else:
                st.session_state["OpenAvatarImg"] = "assistant"
            aa_temperature = st.selectbox(label="`2. Temperature (0~1)`",
                                                  options=["0", "0.2", "0.4", "0.6", "0.8", "1.0"],
                                                  index=1,
                                                  on_change=set_reload_flag)
            if "16k" in aa_llm_model:
                aa_max_resp_max_val = 16 * 1024
            else:
                aa_max_resp_max_val = 4096
            aa_max_resp = st.slider(label="`3. Max response`",
                                            min_value=256,
                                            max_value=aa_max_resp_max_val,
                                            value=512,
                                            on_change=set_reload_flag)
            aa_context_msg = st.select_slider(label="`4. Context message`",
                                                      options=[1, 5, 10, 20],
                                                      value=5,
                                                      on_change=set_reload_flag
                                                      )
            aa_enable_reasoning = st.checkbox(label="`5. Enable Reasoning`",
                                              value=False,
                                              on_change=set_reload_flag
                                              )

        if st.session_state["OpenChatReloadMode"] == True:
            system_prompt = chatbot.select_chat_mode(aa_chat_mode)
            st.session_state["OpenChatReloadMode"] = False

            # initial the greeting
            initial_msg = "I'm OpenChatBot, How may I help you?"
            st.session_state["OpenChatMessages"] = [
                {"role": "system", "content": system_prompt}
            ]
            st.session_state["OpenChatMessagesDisplay"] = [
                {"role": "system", "content": system_prompt}
            ]
        if st.session_state["OpenChatReloadFlag"] == True:
            if "FreeChatSetting" not in st.session_state:
                st.session_state["FreeChatSetting"] = {}
            st.session_state["FreeChatSetting"] = {"model": aa_llm_model, "max_tokens": aa_max_resp,
                                                   "temperature": float(aa_temperature), "context_msg": aa_context_msg,
                                                   "enable_reasoning": aa_enable_reasoning}
            st.session_state["OpenChatReloadFlag"] = False

    # Display chat messages from history on app rerun
    display_message(st.session_state["OpenChatMessagesDisplay"])

    # Accept user input
    if prompt := st.chat_input("Type you input here"):
        # Add user message to chat history
        max_cnt = st.session_state["FreeChatSetting"]["context_msg"]
        st.session_state["OpenChatMessages"] = chatbot.control_msg_history_szie(st.session_state["OpenChatMessages"], max_cnt)
        st.session_state["OpenChatMessages"].append({"role": "user", "content": prompt})
        st.session_state["OpenChatMessagesDisplay"].append({"role": "user", "content": prompt})

        print("HUMAN: " + prompt)
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant", avatar=st.session_state["OpenAvatarImg"]):
            if aa_enable_reasoning:
                with st.status("Thinking...", expanded=True) as status:
                    thinking_placeholder = st.empty()
            message_placeholder = st.empty()
            message_answer = ""
            message_thinking = ""
            with st.spinner("preparing answer"):
                try:
                    async for json_chunk in call_openrouter(
                        model=st.session_state["FreeChatSetting"]["model"],
                        messages=st.session_state["OpenChatMessages"],
                        max_tokens=st.session_state["FreeChatSetting"]["max_tokens"],
                        temperature=st.session_state["FreeChatSetting"]["temperature"],
                        enable_reasoning=st.session_state["FreeChatSetting"]["enable_reasoning"]
                    ):
                        if 'choices' in json_chunk and json_chunk['choices']:
                            delta = json_chunk['choices'][0].get('delta', {})
                            if "reasoning" in delta.keys():
                                if delta["reasoning"]:
                                    message_thinking += delta["reasoning"]
                                    # with thinking_status.status("Thinking...", expanded=True) as status:
                                    thinking_placeholder.markdown(message_thinking)
                            if delta["content"]:
                                message_answer += delta["content"]
                                message_placeholder.markdown(message_answer)
                except Exception as e:
                    print("Error found: ")
                    print(e)
                    print(st.session_state["OpenChatMessages"])
                    st.error(e)
                    if "error" in json_chunk:
                        st.error(json_chunk)
                    st.session_state["OpenChatMessages"].pop(-1)
                    st.session_state["OpenChatMessagesDisplay"].pop(-1)
            if aa_enable_reasoning:
                status.update(label='Thinking complete', state='complete', expanded=False)
            st.session_state["OpenChatMessagesDisplay"].append({"role": "assistant", "content": {"answers": message_answer, "thinking": message_thinking}})
            st.session_state["OpenChatMessages"].append({"role": "assistant", "content": message_answer})
            print("AI: " + message_answer)
            # if aa_voice_name != "None":
            #     chatbot.text_2_speech(full_response, aa_voice_name)
            # btn_placeholder.button(label="Play", key="current", on_click=chatbot.text_2_speech,
            #                        args=(full_response, aa_voice_name,))


if __name__ == "__main__":
    asyncio.run(main())
