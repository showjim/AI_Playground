import os
import shutil
from src.chat import ChatBot

def llm_chat(query_str, docs_path, index_path, env_path = "./"):
    try:
        chat = ChatBot(docs_path, index_path, env_path)
        chat.setup()
        resp = chat.chat(query_str)
        # print(resp)
        return resp
    except Exception as e:
        print(e.__str__())

def llm_chat_langchain(query_str, docs_path, index_path, env_path = "./"):
    try:
        chat = ChatBot(docs_path, index_path, env_path)
        chat.setup_langchain()
        resp = chat.chat_langchain(query_str)
        # print(resp)
        return resp
    except Exception as e:
        print(e.__str__())

if __name__ == '__main__':
    print(llm_chat("output", "../index"))