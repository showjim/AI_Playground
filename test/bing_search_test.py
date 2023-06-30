import json
import os
import shutil
from dotenv import load_dotenv
from langchain.utilities import BingSearchAPIWrapper
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st

shutil.copyfile(os.path.join("../.", "key.txt"), ".env")
load_dotenv()
if os.path.exists(os.path.join("../.", r'config.json')):
    with open(os.path.join("../.", r'config.json')) as config_file:
        config_details = json.load(config_file)
os.environ["BING_SUBSCRIPTION_KEY"] = os.getenv("BING_SUBSCRIPTION_KEY")
os.environ["BING_SEARCH_URL"] = config_details['BING_SEARCH_URL']

search = BingSearchAPIWrapper()
result = search.results("apple list dict", 3)
print(result)