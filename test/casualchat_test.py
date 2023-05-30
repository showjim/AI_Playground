from src.chat import CasualChatBot
import os

env_path = os.path.abspath('../')
casual_chat_bot = CasualChatBot(env_path)
casual_chat_bot.setup_langchain()

def generate_response(prompt):
    # chatbot = hugchat.ChatBot()
    response = casual_chat_bot.chat_langchain(prompt) #chatbot.chat(prompt)
    # print(response)
    return response

while True:
    prompt = input()
    resp = generate_response(prompt)
    # resp = casual_chat_bot.chat_langchain(prompt)
    print(resp)