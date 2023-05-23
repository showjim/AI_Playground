import os
import shutil
from llm.chat import ChatBot

def llm_chat(query_str, docs_path, index_path, env_path = "./"):
    # query_str = """
    # You are a helpful assistant to do meeting record.
    # Please summary this meeting record.
    # Please try to focus on the below requests, and use the bullet format to output the answers for each request:
    # 1. who attend the meeting?
    # 2. Identify key decisions in the transcript.
    # 3. What are the key action items in the meeting?
    # 4. what are the next steps?
    # """
    try:
        chat = ChatBot(docs_path, index_path, env_path)
        chat.setup()
        resp = chat.chat(query_str)
        # print(resp)
        return resp
    except Exception as e:
        print(e.__str__())

def llm_chat_langchain(query_str, docs_path, index_path, env_path = "./"):
    # query_str = """
    #     You are a helpful assistant to do meeting record.
    #     Please summary this meeting record.
    #     Please try to focus on the below requests, and use the bullet format to output the answers for each request:
    #     1. who attend the meeting?
    #     2. Identify key decisions in the transcript.
    #     3. What are the key action items in the meeting?
    #     4. what are the next steps?
    #     """
    try:
        chat = ChatBot(docs_path, index_path, env_path)
        chat.setup_langchain()
        resp = chat.chat_langchain(query_str)
        # print(resp)
        return resp
    except Exception as e:
        print(e.__str__())

if __name__ == '__main__':
    print(llm_chat("output", "index"))