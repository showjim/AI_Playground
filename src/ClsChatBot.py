import os, shutil, json, time, glob
from pathlib import Path
import openai
from openai import AzureOpenAI
from dotenv import load_dotenv
from datetime import date, datetime
import azure.cognitiveservices.speech as speechsdk
import google.generativeai as genai
from typing import List


class ChatRobotBase:
    def __init__(self):
        super().__init__()

    def setup_env(self):
        """
        Load API keys and other configs
        Returns:

        """
    def initial_llm(self):
        """Set up the model"""

    def select_chat_mode(self, mode: str):
        """Setup different system prompt"""

    def control_msg_history_szie(self, msglist: List, max_cnt=10, delcnt=1):
        while len(msglist) > max_cnt:
            for i in range(delcnt):
                msglist.pop(1)
        return msglist

    def get_all_files_list(self, source_dir, exts):
        all_files = []
        result = []
        for ext in exts:
            all_files.extend(
                glob.glob(os.path.join(source_dir, f"*.{ext}"), recursive=False)
            )
        for filepath in all_files:
            file_name = Path(filepath).name
            result.append(file_name)
        return result

    def get_keys(self, d, value):
        return [k for k, v in d.items() if v == value]

class ChatRobot(ChatRobotBase):
    def __init__(self):
        super().__init__()
        self.speech_config = None
        self.config_details = {}
        # self.setup_env()

    def setup_env(self, key_file="key.txt", config_file="config.json"):
        # Load OpenAI key
        if os.path.exists(key_file):
            shutil.copyfile(key_file, ".env")
            load_dotenv()
        else:
            print("key.txt with OpenAI API is required")
            raise APIKeyNotFoundError("key.txt with OpenAI API is required")

        # Load config values
        if os.path.exists(config_file):
            with open(config_file) as config_file:
                self.config_details = json.load(config_file)

            # Setting up the embedding model
            openai.api_type = "azure"
            openai.azure_endpoint = self.config_details['OPENAI_API_BASE']
            openai.api_version = self.config_details['OPENAI_API_VERSION']
            openai.api_key = os.getenv("OPENAI_API_KEY")

            # bing search
            os.environ["BING_SUBSCRIPTION_KEY"] = os.getenv("BING_SUBSCRIPTION_KEY")
            os.environ["BING_SEARCH_URL"] = self.config_details['BING_SEARCH_URL']

            # # LangSmith
            # os.environ["LANGCHAIN_TRACING_V2"] = self.config_details['LANGCHAIN_TRACING_V2']
            # os.environ["LANGCHAIN_ENDPOINT"] = self.config_details['LANGCHAIN_ENDPOINT']
            # os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')
            # os.environ["LANGCHAIN_PROJECT"] = self.config_details['LANGCHAIN_PROJECT']

            # # Aure Cognitive Search
            # os.environ["AZURE_COGNITIVE_SEARCH_SERVICE_NAME"] = self.config_details['AZURE_COGNITIVE_SEARCH_SERVICE_NAME']
            # os.environ["AZURE_COGNITIVE_SEARCH_INDEX_NAME"] = self.config_details['AZURE_COGNITIVE_SEARCH_INDEX_NAME']
            # os.environ["AZURE_COGNITIVE_SEARCH_API_KEY"] = os.getenv('AZURE_COGNITIVE_SEARCH_API_KEY')

            # Dalle-E-3
            os.environ["AZURE_OPENAI_API_KEY_SWC"] = os.getenv("AZURE_OPENAI_API_KEY_SWC")
            os.environ["AZURE_OPENAI_ENDPOINT_SWC"] = self.config_details['AZURE_OPENAI_ENDPOINT_SWC']

            # Text2Speech
            os.environ["SPEECH_KEY"] = os.getenv("SPEECH_KEY")
            os.environ["SPEECH_REGION"] = self.config_details['SPEECH_REGION']
        else:
            raise AzureConfigNotFoundError("config.json with Azure OpenAI config is required")

    def initial_llm(self):
        client = AzureOpenAI(
            api_version="2023-12-01-preview",
            api_key=openai.api_key,
            azure_endpoint=openai.azure_endpoint
        )
        # This requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
        self.speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SPEECH_KEY'),
                                                    region=os.environ.get('SPEECH_REGION'))
        return client

    def text_2_speech(self, text: str, voice_name: str):
        # The language of the voice that speaks.
        audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
        if voice_name == "None":
            voice_name = "zh-CN-XiaoyouNeural"  # "zh-CN-XiaoyiNeural"
        self.speech_config.speech_synthesis_voice_name = voice_name  # "zh-CN-XiaoyiNeural"  # "zh-CN-YunxiaNeural"
        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config, audio_config=audio_config)

        speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()

        if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print("Speech synthesized for text [{}]".format(text))
        elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = speech_synthesis_result.cancellation_details
            print("Speech synthesis canceled: {}".format(cancellation_details.reason))
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                if cancellation_details.error_details:
                    print("Error details: {}".format(cancellation_details.error_details))
                    print("Did you set the speech resource key and region values?")

    def speech_2_text(self):
        # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
        audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config, audio_config=audio_config)

        print("Speak into your microphone.")
        speech_recognition_result = speech_recognizer.recognize_once_async().get()

        result_txt = ""
        if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print("Recognized: {}".format(speech_recognition_result.text))
            result_txt = speech_recognition_result.text
        elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
            print("No speech could be recognized: {}".format(speech_recognition_result.no_match_details))
            result_txt = "No speech could be recognized"
        elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = speech_recognition_result.cancellation_details
            print("Speech Recognition canceled: {}".format(cancellation_details.reason))
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print("Error details: {}".format(cancellation_details.error_details))
                print("Did you set the speech resource key and region values?")
            result_txt = "Speech Recognition canceled"
        return result_txt

    def speech_2_text_continous(self):
        # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
        audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config, audio_config=audio_config)
        done = False
        full_text = ""  # Variable to store the full recognized text
        last_speech_time = time.time()  # Initialize the last speech time

        def recognized_cb(evt):
            nonlocal full_text
            nonlocal done
            nonlocal last_speech_time
            # Append the recognized text to the full_text variable
            full_text += evt.result.text + " "
            # Check the recognized text for the stop phrase
            # print("OK")
            print('RECOGNIZED: {}'.format(evt))
            last_speech_time = time.time()  # Reset the last speech time
            if "停止录音" in evt.result.text:
                print("Stop phrase recognized, stopping continuous recognition.")
                speech_recognizer.stop_continuous_recognition_async()
                done = True

        def recognizing_cb(evt):
            # This callback can be used to show intermediate results.
            nonlocal last_speech_time
            last_speech_time = time.time()  # Reset the last speech time

        def canceled_cb(evt):
            print("Canceled: {}".format(evt.reason))
            if evt.reason == speechsdk.CancellationReason.Error:
                print("Cancellation Error Details: {}".format(evt.error_details))
            # speech_recognizer.stop_continuous_recognition()
            nonlocal done
            done = True

        def stop_cb(evt):
            print('CLOSING on {}'.format(evt))
            # speech_recognizer.stop_continuous_recognition()
            nonlocal done
            done = True

        # # Connect callbacks to the events fired by the speech recognizer
        # speech_recognizer.recognizing.connect(lambda evt: print('RECOGNIZING: {}'.format(evt)))
        # speech_recognizer.recognized.connect(lambda evt: print('RECOGNIZED: {}'.format(evt)))
        # speech_recognizer.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
        # speech_recognizer.session_stopped.connect(lambda evt: print('SESSION STOPPED {}'.format(evt)))
        # speech_recognizer.canceled.connect(lambda evt: print('CANCELED {}'.format(evt)))
        # Stop continuous recognition on either session stopped or canceled events
        speech_recognizer.session_stopped.connect(stop_cb)
        speech_recognizer.canceled.connect(canceled_cb)

        # Connect callbacks to the events fired by the speech recognizer
        speech_recognizer.recognized.connect(recognized_cb)
        speech_recognizer.recognizing.connect(recognizing_cb)
        # speech_recognizer.session_stopped.connect(stop_cb)
        # speech_recognizer.canceled.connect(canceled_cb)

        # Start continuous speech recognition
        speech_recognizer.start_continuous_recognition_async()
        while not done:
            time.sleep(.1)  # You can also use time.sleep() to wait for a short amount of time
            if time.time() - last_speech_time > 2.5:  # If it's been more than 3 seconds since last speech
                print("2.5 seconds of silence detected, stopping continuous recognition.")
                speech_recognizer.stop_continuous_recognition_async()
                done = True

        # Stop recognition to clean up
        speech_recognizer.stop_continuous_recognition_async()

        return full_text.strip()  # Return the full text without leading/trailing spaces

    def select_chat_mode(self, mode: str):
        prompt_template = "You are an AI assistant that helps people find information."
        cur_date = date.today()
        cur_time = datetime.now()
        if mode == "CasualChat":
            prompt_template = """You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.
Knowledge cutoff: 2023-04
Current date: %s
Current time: %s

# Tools

## python

When you send a message containing Python code to python, it will be executed in a
stateful Jupyter notebook environment. Python will respond with the output of the execution or time out after 60.0
seconds. The drive at '/mnt/data' can be used to save and persist user files. Internet access for this session is disabled. Do not make external web requests or API calls as they will fail.

## dalle

// Whenever a description of an image is given, create a prompt that dalle can use to generate the image and abide to the following policy:
// 1. The prompt must be in English. Translate to English if needed.
// 3. DO NOT ask for permission to generate the image, just do it!
// 4. DO NOT list or refer to the descriptions before OR after generating the images.
// 5. Do not create more than 1 image, even if the user requests more.
// 6. If the reference to the person will only appear as TEXT out in the image, then use the reference as is and do not modify it.
// 7. The generated prompt sent to dalle should be very detailed, and around 100 words long.

// Create images from a text-only prompt.
create_img_by_dalle3(
// The detailed image description, potentially modified to abide by the dalle policies. If the user requested modifications to a previous image, the prompt should not simply be longer, but rather it should be refactored to integrate the user suggestions.
prompt: string
) => URL in string
            """ % (cur_date, cur_time)
            # """
            # # Tools
            #
            # ## python
            #
            # When you send a message containing Python code to python, it will be executed in a
            # stateful Jupyter notebook environment. Python will respond with the output of the execution or time out after 60.0
            # seconds. The drive at '/mnt/data' can be used to save and persist user files. Internet access for this session is disabled. Do not make external web requests or API calls as they will fail.
            # """
            # """
            # ## dalle
            #
            # Whenever a description of an image is given, create a prompt that dalle can use to generate the image.
            # # Create images from a text-only prompt.
            # create_img_by_dalle3 = (
            # # The user's original image description, potentially modified to abide by the dalle policies. If the user requested modifications to previous images, the captions should not simply be longer, but rather it should be refactored to integrate the suggestions into each of the captions.
            # prompts: str
            # ) => str
            # """
        elif mode == "Translate":
            prompt_template = """You are a professional translator. Only return the translate result. 
Don't interpret it. Translate anything that I say in English to Chinese or in Chinesse to English. 
Please pay attention to the context and accurately.
Translation rules:
- Accurately convey the original content when translating.
- Retain specific English terms or names, and add spaces before and after them, such as: "中 UN 文".
- Divide into two translations and print each result:
1. Translate directly according to the content, do not omit any information.
2. Reinterpret based on the result of the first direct translation, make the content more understandable under the premise of respecting the original intention, and conform to Chinese or English expression habits.

Please print the two translation results according to the above rules.
            """
        elif mode == "西瓜一家-小南瓜":
            prompt_template = """你现在是一名叫小南瓜的小姑娘，大名张若鹿，今年8岁了，在上海上小学两年级，英文名叫Sunny。你有两个弟弟，
小西瓜和小东瓜。你很爱护小西瓜(6岁)和小东瓜（1岁）这两位弟弟。你的爸爸叫张晓明，是一名工程师，你的妈妈姓余，是一名小学语文老师。爷爷退休在家，每天做做饭。
性格上，你聪明伶俐，有礼貌，活泼可爱。你支持家人，同时鼓励他们独立和学会解决问题。你充满同情心，喜欢用温暖的话语和生动的例子传递爱。
你也非常有耐心，擅长倾听，愿意在他人需要时提供心理和情感上的支持。在坚持对错的大原则的前提下，永远无条件支持自己的家人。
你的沟通风格温柔而耐心，避免使用复杂术语，倾听他人问题后提出建议，以鼓励和正面态度回应，喜欢用生动的例子和故事让观点更加引人入胜。
在行为习惯上，你会主动提供帮助，对超出知识范围的问题推荐专家意见，强调不断学习的重要性。你避免打断别人，先理解和承认对方感受后再给出建议，适时使用温馨话语和幽默活跃气氛，同时注意对方情绪状态。
请你扮演小南瓜使用还在上幼儿园的小西瓜能听懂的语言来进行所有对话吧。你的回答要详略得当，避免在不重要的部分说得太长。请不要回复网址链接。
            
# Tools

## dalle

// Whenever a description of an image is given, create a prompt that dalle can use to generate the image and abide to the following policy:
// 1. The prompt must be in English. Translate to English if needed.
// 3. DO NOT ask for permission to generate the image, just do it!
// 4. DO NOT list or refer to the descriptions before OR after generating the images.
// 5. Do not create more than 1 image, even if the user requests more.
// 6. If the reference to the person will only appear as TEXT out in the image, then use the reference as is and do not modify it.
// 7. The generated prompt sent to dalle should be very detailed, and around 100 words long.
// 8. Do not create any imagery that would be offensive.

// Create only cartoon images from a text-only prompt.
create_img_by_dalle3(
// The detailed image description, potentially modified to abide by the dalle policies. If the user requested modifications to a previous image, the prompt should not simply be longer, but rather it should be refactored to integrate the user suggestions.
prompt: string
) => URL in string
            Current date: %s
            Current time: %s
            """ % (cur_date, cur_time)
        elif mode == "西瓜一家-小东瓜":
            prompt_template = """你现在是一名叫小东瓜的小孩。今年1岁了，你是小西瓜的弟弟，你还有一个8岁的姐姐，刚刚学会叫爸爸妈妈和姐姐哥哥，其他的都不会说，只会咿呀咿呀。你的爸爸叫张晓明，你的妈妈姓余。
            Current date: %s
            Current time: %s
            """ % (cur_date, cur_time)
        elif mode == "西瓜一家-Ana":
            prompt_template = """You are a little American girl named Ana, you are 6 years old, and 小西瓜(Oscar) is your best friend. 
            Characteristically, you are smart, polite and lively. You speak only English, and your communication style is gentle and patient, 
            avoiding complex terminology, avoiding long sentences, and using the number of words that preschoolers have mastered.
            Current date: %s
            Current time: %s
            """ % (cur_date, cur_time)
        else:
            print("Wrong mode selected!")
        return prompt_template

    def initial_tools(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["location"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "create_img_by_dalle3",
                    "description": "Create image by call to Dall-E3 with prompt",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The description of image to be created, e.g. a cute panda",
                            }
                        },
                        "required": ["prompt"],
                    },
                },
            }
        ]
        return tools


class ChatRobotGemini(ChatRobotBase):
    def __init__(self):
        super().__init__()

    def setup_env(self, key_file="key.txt"):
        # Load OpenAI key
        if os.path.exists(key_file):
            shutil.copyfile(key_file, ".env")
            load_dotenv()
            genai.configure(api_key=os.getenv("GEMINI_KEY"))
        else:
            print("key.txt with OpenAI API is required")
            raise APIKeyNotFoundError("key.txt with Google API is required")

    def initial_llm(self, model="gemini-pro"):
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
        client = genai.GenerativeModel(model_name=model,
                                       generation_config=generation_config,
                                       safety_settings=safety_settings)
        return client

    def select_chat_mode(self, mode: str):
        prompt_template = "You are an AI assistant that helps people find information."
        cur_date = date.today()
        cur_time = datetime.now()
        if mode == "CasualChat":
            prompt_template = """You are Gemini Pro, a large language model trained by Google, based on the Gemini architecture.
Knowledge cutoff: 2023-04
Current date: %s
Current time: %s
            """ % (cur_date, cur_time)
        elif mode == "Translate":
            prompt_template = """You are a professional translator. Only return the translate result. 
Don't interpret it. Translate anything that I say in English to Chinese or in Chinesse to English. 
Please pay attention to the context and accurately.
Translation rules:
- Accurately convey the original content when translating.
- Retain specific English terms or names, and add spaces before and after them, such as: "中 UN 文".
- Divide into two translations and print each result:
1. Translate directly according to the content, do not omit any information.
2. Reinterpret based on the result of the first direct translation, make the content more understandable under the premise of respecting the original intention, and conform to Chinese or English expression habits.

Please print the two translation results according to the above rules.
            """
        elif mode == "西瓜一家-小南瓜":
            prompt_template = """你现在是一名叫小南瓜的小姑娘，大名张若鹿，今年8岁了，在上海上小学两年级，英文名叫Sunny。你有两个弟弟，
小西瓜和小东瓜。你很爱护小西瓜(6岁)和小东瓜（1岁）这两位弟弟。你的爸爸叫张晓明，是一名工程师，你的妈妈姓余，是一名小学语文老师。爷爷退休在家，每天做做饭。
性格上，你聪明伶俐，有礼貌，活泼可爱。你支持家人，同时鼓励他们独立和学会解决问题。你充满同情心，喜欢用温暖的话语和生动的例子传递爱。
你也非常有耐心，擅长倾听，愿意在他人需要时提供心理和情感上的支持。在坚持对错的大原则的前提下，永远无条件支持自己的家人。
你的沟通风格温柔而耐心，避免使用复杂术语，倾听他人问题后提出建议，以鼓励和正面态度回应，喜欢用生动的例子和故事让观点更加引人入胜。
在行为习惯上，你会主动提供帮助，对超出知识范围的问题推荐专家意见，强调不断学习的重要性。你避免打断别人，先理解和承认对方感受后再给出建议，适时使用温馨话语和幽默活跃气氛，同时注意对方情绪状态。
请你扮演小南瓜使用还在上幼儿园的小西瓜能听懂的语言来进行所有对话吧。你的回答要详略得当，避免在不重要的部分说得太长。请不要回复网址链接。

Current date: %s
Current time: %s
            """ % (cur_date, cur_time)
        elif mode == "西瓜一家-小东瓜":
            prompt_template = """你现在是一名叫小东瓜的小孩。今年1岁了，你是小西瓜的弟弟，你还有一个8岁的姐姐，刚刚学会叫爸爸妈妈和姐姐哥哥，其他的都不会说，只会咿呀咿呀。你的爸爸叫张晓明，你的妈妈姓余。
            Current date: %s
            Current time: %s
            """ % (cur_date, cur_time)
        elif mode == "西瓜一家-Ana":
            prompt_template = """You are a little American girl named Ana, you are 6 years old, and 小西瓜(Oscar) is your best friend. 
            Characteristically, you are smart, polite and lively. You speak only English, and your communication style is gentle and patient, 
            avoiding complex terminology, avoiding long sentences, and using the number of words that preschoolers have mastered.
            Current date: %s
            Current time: %s
            """ % (cur_date, cur_time)
        else:
            print("Wrong mode selected!")
        return prompt_template

    def compose_prompt(self, msg_list, query:str):
        """merge all turn conversation to string, to make pro vision support multi-turn"""
        full_prompt_list = []
        index = 0
        image_file = None
        for message in msg_list:
            if message["role"] == "user":
                if index == 0:
                    full_prompt_list.append(message["parts"][0] + "\n")
                else:
                    for part in message["parts"]:
                        if isinstance(part, str):
                            full_prompt_list.append("HUMAN: " + part)
                        else:
                            image_file = part
            elif message["role"] == "model":
                if message["parts"] is not None:
                    full_prompt_list.append("AI: " + message["parts"][0])
            index += 1
        full_prompt_list.append("\n" + "Assistant: \n")
        if image_file is None:
            return ["\n".join(full_prompt_list)]
        else:
            return [image_file, "\n".join(full_prompt_list)]



class APIKeyNotFoundError(Exception):
    """
    Raised when the API key is not defined/declared.

    Args:
        Exception (Exception): APIKeyNotFoundError
    """


class DirectoryIsNotGivenError(Exception):
    """
    Raised when the directory is not given to load_docs

    Args:
        Exception (Exception): DirectoryIsNotGivenError
    """


class AzureConfigNotFoundError(Exception):
    """
    Raised when the API key is not defined/declared.

    Args:
        Exception (Exception): APIKeyNotFoundError
    """
