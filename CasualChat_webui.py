import streamlit as st
from streamlit_chat import message
from src.chat import CasualChatBot
import os

__version__ = "Beta V0.0.2"

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


# User input
## Function for taking user provided prompt as input
def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text


def main():
    # Sidebar contents
    with st.sidebar:
        st.title('🤗💬 Casual Chat Web-UI App(Inside TER)')
        st.markdown('''
        ## About
        This app is an Azure OpenAI-powered chatbot built using:
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/en/latest/)
    
        💡 Note: No API key required!
        ''')
        # add_vertical_space(5)
        st.write('Made by Jerry Zhou')
        st.markdown('''
                - [Source Code](https://github.com/showjim/AutoMeetingMinutes/)
                ''')

    # Generate empty lists for chain, generated and past.
    ## generated stores langchain chain
    if "chain" not in st.session_state:
        chain = casual_chat_bot.setup_langchain()
        st.session_state["chain"] = chain
    ## generated stores AI generated responses
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["I'm CasualChat, How may I help you?"]
    ## past stores User's questions
    if 'past' not in st.session_state:
        st.session_state['past'] = ['Hi!']

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
            st.session_state.past.append(user_input)
            st.session_state.generated.append(response)

        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))

if __name__ == "__main__":
    main()