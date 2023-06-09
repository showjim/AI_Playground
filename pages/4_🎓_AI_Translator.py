import streamlit as st
from streamlit_chat import message
from src.chat import CasualChatBot
import os

# __version__ = "Beta V0.0.2"

env_path = os.path.abspath('.')
casual_chat_bot = CasualChatBot(env_path)
# casual_chat_bot.setup_langchain() #change to st, then we can use memory function

st.set_page_config(page_title="AI Translator based on OpenAI")

# Response output
## Function for taking user prompt as input followed by producing AI generated responses
def generate_response(prompt):
    # casual_chat_bot.chat_langchain(prompt) #chatbot.chat(prompt)
    # change to here, call predict from st.session_state["chain"] to use memory function from langchain
    response = st.session_state["T_chain"].predict(human_input=prompt)
    # print(response)
    return response


# User input
## Function for taking user provided prompt as input
def get_text():
    # input_text = st.text_input("You: ", "", key="input")
    input_text = st.text_area("You: ", "", key="input")
    return input_text


def main():
    # Generate empty lists for chain, generated and past.
    ## generated stores langchain chain
    if "T_chain" not in st.session_state:
        chain = casual_chat_bot.initial_llm("Translate")
        st.session_state["T_chain"] = chain
    ## generated stores AI generated responses
    if 'result' not in st.session_state:
        st.session_state['result'] = ["I'm AI Translator, How may I help you?"]
    ## past stores User's questions
    if 'original' not in st.session_state:
        st.session_state['original'] = ['Hi!']

    # Layout of input/response containers
    response_container = st.container()
    # colored_header(label='', description='', color_name='blue-30')
    input_container = st.container()


    ## Applying the user input box
    with input_container:
        user_input = get_text()


    ## Conditional display of AI generated responses as a function of user provided prompts
    with response_container:
        if user_input:
            response = generate_response(user_input)
            st.session_state.original.append(user_input)
            st.session_state.result.append(response)

        if st.session_state['result']:
            for i in range(len(st.session_state['result'])):
                message(st.session_state['original'][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["result"][i], key=str(i))

if __name__ == "__main__":
    main()