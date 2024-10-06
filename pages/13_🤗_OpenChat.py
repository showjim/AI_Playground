import hmac

import streamlit as st
import os, time, json, io
from src.ClsChatBot import ChatRobotOpenRouter


__version__ = "Beta V0.0.4"
env_path = os.path.abspath(".")

chatbot = ChatRobotOpenRouter()
chatbot.setup_env()
client = chatbot.initial_llm()


st.set_page_config(page_title="OpenChat - Chatbot With Variable LLMs")


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

def main():
    index = 0
    st.title("🤗Open Chat Web-UI App " + __version__)
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
                                                options=["CasualChat", "Translate"],
                                                index=0,
                                                on_change=set_reload_mode)
            aa_llm_model = st.selectbox(label="`1. LLM Model`",
                                        options=["openchat/openchat-7b:free",
                                                 "meta-llama/llama-3.2-11b-vision-instruct:free",
                                                 "meta-llama/llama-3.2-90b-vision-instruct",
                                                 "mistralai/mixtral-8x22b-instruct",
                                                 "qwen/qwen-2.5-72b-instruct",
                                                 "deepseek/deepseek-chat",
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

        if st.session_state["OpenChatReloadMode"] == True:
            system_prompt = chatbot.select_chat_mode(aa_chat_mode)
            st.session_state["OpenChatReloadMode"] = False

            # initial the greeting
            initial_msg = "I'm OpenChatBot, How may I help you?"

            st.session_state["OpenChatMessages"] = [
                {"role": "system", "content": system_prompt},
                {"role": "assistant", "content": initial_msg}
            ]
            st.session_state["OpenChatMessagesDisplay"] = [
                {"role": "system", "content": system_prompt},
                {"role": "assistant", "content": initial_msg}
            ]
        if st.session_state["OpenChatReloadFlag"] == True:
            if "FreeChatSetting" not in st.session_state:
                st.session_state["FreeChatSetting"] = {}
            st.session_state["FreeChatSetting"] = {"model": aa_llm_model, "max_tokens": aa_max_resp,
                                                   "temperature": float(aa_temperature), "context_msg": aa_context_msg}
            st.session_state["OpenChatReloadFlag"] = False

    # Display chat messages from history on app rerun
    for message in st.session_state["OpenChatMessagesDisplay"]:
        if message["role"] == "user":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        elif message["role"] == "assistant":
            if message["content"] is not None:
                with st.chat_message(name=message["role"], avatar=st.session_state["OpenAvatarImg"]):
                    st.markdown(message["content"])
                    # st.button(label="Play", key="history" + str(index), on_click=chatbot.text_2_speech,
                    #           args=(message["content"], aa_voice_name,))
                    index += 1
                    if "image" in message.keys():
                        st.image(message["image"], width=256)

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
            message_placeholder = st.empty()
            btn_placeholder = st.empty()
            full_response = ""
            function_response = ""
            with st.spinner("preparing answer"):
                try:
                    response = client.chat.completions.create(
                        model=st.session_state["FreeChatSetting"]["model"],
                        messages=st.session_state["OpenChatMessages"],
                        max_tokens=st.session_state["FreeChatSetting"]["max_tokens"],
                        # default max tokens is low so set higher
                        temperature=st.session_state["FreeChatSetting"]["temperature"],
                        stream=True,
                    )
                    for chunk in response:
                        # process normal response and tool_calls response
                        if len(chunk.choices) > 0:
                            deltas = chunk.choices[0].delta
                            if deltas.content is not None:
                                full_response += deltas.content  # ["answer"]  # .choices[0].delta.get("content", "")
                                time.sleep(0.001)
                                message_placeholder.markdown(full_response + "▌")
                except Exception as e:
                    print("Error found: ")
                    print(e)
                    print(st.session_state["OpenChatMessages"])
                    st.error(e)
                    st.session_state["OpenChatMessages"].pop(-1)
                    st.session_state["OpenChatMessagesDisplay"].pop(-1)


                full_response = full_response
            message_placeholder.markdown(full_response)

            st.session_state["OpenChatMessages"].append({"role": "assistant", "content": full_response})
            if function_response.startswith("https://"):
                st.session_state["OpenChatMessagesDisplay"].append(
                    {"role": "assistant", "content": full_response, "image": function_response})
                st.image(function_response)
            else:
                st.session_state["OpenChatMessagesDisplay"].append(
                    {"role": "assistant", "content": full_response})
            print("AI: " + full_response)
            # if aa_voice_name != "None":
            #     chatbot.text_2_speech(full_response, aa_voice_name)
            # btn_placeholder.button(label="Play", key="current", on_click=chatbot.text_2_speech,
            #                        args=(full_response, aa_voice_name,))


if __name__ == "__main__":
    main()
