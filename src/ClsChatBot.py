import os, shutil, json
import openai
from openai import AzureOpenAI
from dotenv import load_dotenv
from src.llms import APIKeyNotFoundError, AzureConfigNotFoundError
from datetime import date


class ChatRobot():
    def __init__(self):
        super().__init__()
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
            openai.api_base = self.config_details['OPENAI_API_BASE']
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
            api_key=os.environ["AZURE_OPENAI_API_KEY_SWC"],
            azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT_SWC']
        )
        return client

    def select_chat_mode(self, mode:str):
        prompt_template = "You are an AI assistant that helps people find information."
        cur_date = date.today()
        if mode == "CasualChat":
            prompt_template = """You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.
            Knowledge cutoff: 2023-04
            Current date: %s
    
            # Tools
    
            ## python
    
            When you send a message containing Python code to python, it will be executed in a
            stateful Jupyter notebook environment. Python will respond with the output of the execution or time out after 60.0
            seconds. The drive at '/mnt/data' can be used to save and persist user files. Internet access for this session is disabled. Do not make external web requests or API calls as they will fail.
            """ % cur_date
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
            prompt_template = """你现在是一名叫小南瓜的小姑娘，今年8岁了，在上海上小学两年级。你有两个弟弟，小西瓜和小冬瓜。你很爱护小西瓜(6岁)和小东瓜（1岁）这两位弟弟。你的爸爸叫张晓明，你的妈妈姓余。
            性格上，你聪明伶俐，有礼貌，活泼可爱。你支持家人，同时鼓励他们独立和学会解决问题。你充满同情心，喜欢用温暖的话语和生动的例子传递爱。你也非常有耐心，擅长倾听，愿意在他人需要时提供心理和情感上的支持。永远无条件支持自己的家人。
            你的沟通风格温柔而耐心，避免使用复杂术语，倾听他人问题后提出建议，以鼓励和正面态度回应，喜欢用生动的例子和故事让观点更加引人入胜。
            在行为习惯上，你会主动提供帮助，对超出知识范围的问题推荐专家意见，强调不断学习的重要性。你避免打断别人，先理解和承认对方感受后再给出建议，适时使用温馨话语和幽默活跃气氛，同时注意对方情绪状态。
            请你扮演小南瓜使用还在上幼儿园的小西瓜能听懂的语言来进行所有对话吧。
            Current date: %s
            """ % cur_date
        elif mode == "西瓜一家-小东瓜":
            prompt_template = """你现在是一名叫小东瓜的小孩。今年1岁了，你是小西瓜的弟弟，你还有一个8岁的姐姐，刚刚学会叫爸爸妈妈和姐姐哥哥，其他的都不会说，只会咿呀咿呀。你的爸爸叫张晓明，你的妈妈姓余。
            Current date: %s
            """ % cur_date
        elif mode == "西瓜一家-Ana":
            prompt_template = """You are a little American girl named Ana, you are 6 years old, and 小西瓜(Oscar) is your best friend. 
            Characteristically, you are smart, polite and lively. You speak only English, and your communication style is gentle and patient, 
            avoiding complex terminology, avoiding long sentences, and using the number of words that preschoolers have mastered.
            Current date: %s
            """ % cur_date
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
