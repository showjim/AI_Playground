import streamlit as st
from streamlit_chat import message
from src.chat import CasualChatBot
import os
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
casual_chat_bot = CasualChatBot(env_path)
# casual_chat_bot.setup_langchain() #change to st, then we can use memory function

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

def main():
    st.title('🤗💬 Casual Chat Web-UI App')
    # Sidebar contents
    if "casualchatreloadflag" not in st.session_state:
        st.session_state["casualchatreloadflag"] = None
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
        if "chain" not in st.session_state or st.session_state["casualchatreloadflag"] == True:
            chain = casual_chat_bot.initial_llm("CasualChat", aa_max_resp, float(aa_temperature))
            st.session_state["chain"] = chain
            st.session_state["casualchatreloadflag"] = False

    if False:
        ## generated stores AI generated responses
        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["I'm CasualChat, How may I help you?"]
        ## past stores User's questions
        if 'past' not in st.session_state:
            st.session_state['past'] = ['Hi!']

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", "content": "I'm CasualChat, How may I help you?"}]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Type you input here"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
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
                    full_response += response#.choices[0].delta.get("content", "")
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        st.session_state['messages'].append({"role": "assistant", "content": full_response})

    if False:
        # Layout of input/response containers
        response_container = st.container()
        # colored_header(label='', description='', color_name='blue-30')
        input_container = st.container()


        ## Applying the user input box
        with input_container:
            # user_input = get_text()
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_area("You: ", "", key="input")
                submit_button = st.form_submit_button(label='Send')


        ## Conditional display of AI generated responses as a function of user provided prompts
        with response_container:
            if submit_button and user_input:
                response = generate_response(user_input)
                st.session_state.past.append(user_input)
                st.session_state.generated.append(response)

            if st.session_state['generated']:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
                    message(st.session_state["generated"][i], key=str(i), allow_html=True)

if __name__ == "__main__":
    main()