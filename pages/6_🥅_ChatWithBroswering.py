import streamlit as st
from src.chat import AgentChatBot
import os, time
from langchain.callbacks import StreamlitCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text=initial_text
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # "/" is a marker to show difference
        # you don't need it
        self.text+=token+"/"
        # self.container.markdown(self.text)

# __version__ = "Beta V0.0.2"

env_path = os.path.abspath('.')
casual_chat_bot = AgentChatBot(env_path, "") #CasualChatBot(env_path)
# casual_chat_bot.setup_langchain() #change to st, then we can use memory function

st.set_page_config(page_title="🥅 Chat With Broswering")

def set_reload_flag():
    # st.write("New document need upload")
    st.session_state["B_casualchatreloadflag"] = True

def main():
    st.title('🤗🥅 Chat With Broswering')
    # Sidebar contents
    if "B_casualchatreloadflag" not in st.session_state:
        st.session_state["B_casualchatreloadflag"] = None
    with st.sidebar:
        st.sidebar.expander("Settings")
        st.sidebar.subheader("Parameter for Chatbot")

        aa_temperature = st.sidebar.selectbox(label="1.Temperature (0~1)",
                                              options=["0", "0.2", "0.4", "0.6", "0.8", "1.0"],
                                              index=1,
                                              on_change=set_reload_flag)
        aa_max_resp = st.sidebar.slider(label="2.Max response",
                                        min_value=256,
                                        max_value=2048,
                                        value=512,
                                        on_change=set_reload_flag)
        if "B_chain" not in st.session_state or st.session_state["B_casualchatreloadflag"] == True:
            chain = casual_chat_bot.initial_llm("bing_search", "", int(aa_max_resp), float(aa_temperature))
            st.session_state["B_chain"] = chain
            st.session_state["B_casualchatreloadflag"] = False

    # Initialize chat history
    if "B_messages" not in st.session_state:
        st.session_state['B_messages'] = [{"role": "assistant", "content": "I'm CasualChat, How may I help you?"}]

    # Display chat messages from history on app rerun
    for message in st.session_state["B_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Type you input here"):
        # Add user message to chat history
        st.session_state["B_messages"].append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container()) #StreamingStdOutCallbackHandler StreamlitCallbackHandler(st.container()) StreamHandler(st.container())
            message_placeholder = st.empty()
            full_response = ""

            full_response = st.session_state["B_chain"].run(prompt, callbacks=[st_callback])
            # for response in st.session_state["B_chain"].predict(human_input=prompt, callbacks=[st_callback]):
            #     full_response += response #.choices[0].delta.get("content", "")
            #     time.sleep(0.01)
            #     message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        st.session_state['B_messages'].append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()