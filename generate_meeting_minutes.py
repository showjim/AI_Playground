import os
import shutil
from llm.chat import ChatBot

def llm_chat(docs_path, index_path):
    query_str = """
    You are a helpful assistant to do meeting record.
    Please summary this meeting record.
    Please use the bullet format to output the answer, and try to focus on the below requests: 
    1. who attend the meeting?
    2. Identify key decisions in the transcript.
    3. What are the key action items in the meeting?
    4. what are the next steps?
    """
    try:
        chat = ChatBot(docs_path, index_path)
        resp = chat.chat(query_str)
        print(resp)
    except Exception as e:
        print(e.__str__())

if __name__ == '__main__':
    llm_chat("output", "index")