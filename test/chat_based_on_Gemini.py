"""
At the command line, only need to run once to install the package via pip:

$ pip install google-generativeai
"""

import google.generativeai as genai
import PIL.Image

img = PIL.Image.open(r'./images/generated_image20231124-195404.png')


genai.configure(api_key="Gemini Key")

# Set up the model
generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 512,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
]

model = genai.GenerativeModel(model_name="gemini-pro-vision" , #"gemini-pro-vision", #"gemini-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

messages = [
    {'role':'user', 'parts': ["Briefly explain what I said to a young child."]},
    {'role':'model', 'parts': ["OK"]},
]

chat = model.start_chat(history=messages)

def do_chat():
    while True:
        prompt = input()
        response = chat.send_message(prompt, stream=True)
        full_response = ""
        for chunk in response:
            full_response += chunk.text
        print("AI: ")
        print(full_response)

    for message in chat.history:
      print(f'**{message.role}**: {message.parts[0].text}')

def do_chat_low_level():
    while True:
        prompt = input()
        messages.append({'role': 'user', 'parts': [prompt]})
        response = model.generate_content(messages, stream=True)
        
        full_response = ""
        for chunk in response:
            full_response += chunk.text
        print("AI: ")
        print(full_response)
        messages.append({'role': 'model',
                         'parts': [full_response]})


def do_chat_img():
    while True:
        prompt = input()
        # messages.append({'role': 'user', 'parts': [prompt]})
        response = model.generate_content([prompt, img], stream=True)

        full_response = ""
        for chunk in response:
            full_response += chunk.text
        print("AI: ")
        print(full_response)
        # messages.append({'role': 'model',
        #                  'parts': [full_response]})

# do_chat()
# do_chat_low_level()
do_chat_img()
