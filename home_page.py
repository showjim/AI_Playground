# Contents of ~/my_app/main_page.py
import streamlit as st

__version__ = "Beta V1.2"

st.set_page_config(
    page_title="Jerry's AI Playground",
    page_icon="🛝",
)

st.sidebar.success("Select a page above.")

st.write(
    """
    # Main page 🛝🎈
    This app is an Azure OpenAI-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/en/latest/)

    💡 Note: No API key required!
    
    - [Source Code](https://github.com/showjim/AutoMeetingMinutes/)
    
    Made by Jerry Zhou
    """
)

