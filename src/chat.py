import glob
from datetime import date
import os.path
from pathlib import Path
from src.llms import OpenAI, OpenAIAzure
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory

from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.agents import AgentType, initialize_agent, load_tools, Tool
from langchain.callbacks.base import BaseCallbackHandler

from langchain.chains.qa_with_sources.stuff_prompt import EXAMPLE_PROMPT as DOC_PROMPT
from langchain.chains.qa_with_sources.map_reduce_prompt import QUESTION_PROMPT as QUESTION_PROMPT
from langchain.chains.qa_with_sources.refine_prompts import (
    DEFAULT_TEXT_QA_PROMPT as REFINE_TEXT_QA_PROMPT,
    DEFAULT_REFINE_PROMPT as REFINE_PROMPT,
)
from langchain.chains.question_answering.map_rerank_prompt import PROMPT as map_rerank_prompt

from langchain_community.utilities import BingSearchAPIWrapper

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text=initial_text
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # "/" is a marker to show difference
        # you don't need it
        self.text+=token+"/"
        # self.container.markdown(self.text)

class ChatBot():
    def __init__(self, docs_path:str, index_path:str, env_path:str):
        super().__init__()
        # self.model = OpenAI(dir=env_path)
        # self.model = OpenAIAzure(dir=env_path)
        self.doc_summary_index = None
        self.model = OpenAIAzure(dir=env_path)
        self.model.setup_env()
        self.docs_path = docs_path
        self.index_path =index_path
        # self.service_context = self.model.create_chat_model()
        # self.documents = self.model.load_docs(docs_path)
        # self.model.build_index(self.service_context, self.documents, index_path)
        # self.doc_summary_index = self.model.rebuild_index_from_dir(index_path)
    # def setup(self):
    #     DEFAULT_INDEX_FILE = "index.faiss"
    #     index_file = os.path.join(Path(self.index_path), Path(DEFAULT_INDEX_FILE))
    #     self.service_context = self.model.create_chat_model()
    #     if not os.path.exists(index_file):
    #         self.documents = self.model.load_docs(self.docs_path)
    #         self.model.build_index(self.service_context, self.documents, self.index_path)
    #     self.doc_summary_index = self.model.rebuild_index_from_dir(self.index_path, self.service_context)
    # def chat(self, query_str:str):
    #     query_engine = self.doc_summary_index.as_query_engine(response_mode="tree_summarize")  # , streaming=True)
    #     response = query_engine.query(query_str)
    #     return response.response

    def initial_llm(self,model_name, num_output, temperature):
        self.llm, self.embedding = self.model.create_chat_model(model_name, num_output, temperature)

    def setup_vectordb(self, filname:str):
        DEFAULT_INDEX_FILE = Path(filname).stem + ".faiss"
        index_file = os.path.join(Path(self.index_path), Path(DEFAULT_INDEX_FILE))
        # self.llm, self.embedding = self.model.create_chat_model()
        if not os.path.exists(index_file):
            self.documents = self.model.load_docs(filname) #(self.docs_path)
            self.model.build_index(self.embedding, self.documents, self.index_path, Path(filname).stem)
        self.doc_summary_index = self.model.rebuild_index_from_dir(self.index_path, self.embedding, Path(filname).stem)
        return self.doc_summary_index

    def get_all_files_list(self, source_dir, ext:str = "faiss"):
        all_files = []
        result = []
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"*.{ext}"), recursive=False)
        )
        for filepath in all_files:
            file_basename = Path(filepath).stem
            result.append(file_basename)
        return result

    def load_vectordb(self, all_files):
        self.doc_summary_index = self.model.rebuild_index_by_list(self.index_path, self.embedding, all_files)
        return self.doc_summary_index

    def chat(self, query_str:str, doc_summary_index):
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, please think rationally and answer from your own knowledge base 

        {context}

        Question: {question}
        """
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        chain_type_kwargs = {"prompt": PROMPT}
        qa = RetrievalQA.from_chain_type(llm=self.llm,
                                         chain_type="stuff",
                                         retriever=doc_summary_index.as_retriever(),
                                         verbose=True,
                                         chain_type_kwargs=chain_type_kwargs)
        resp = qa.run(query_str)
        return resp

    def chat_QA(self, doc_summary_index):
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, please think rationally and answer from your own knowledge base. 

        {context}

        Question: {question}
        """
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        chain_type_kwargs = {"prompt": PROMPT}
        # qa = RetrievalQA.from_chain_type(llm=self.llm,
        #                                  chain_type="stuff",
        #                                  retriever=self.doc_summary_index.as_retriever(),
        #                                  verbose=True,
        #                                  chain_type_kwargs=chain_type_kwargs)
        qa_chain = ConversationalRetrievalChain.from_llm(llm=self.llm,
                                                         retriever=doc_summary_index.as_retriever(),
                                                         memory=ConversationBufferMemory(
                                                             memory_key="chat_history",
                                                             input_key='question',
                                                             output_key='answer',
                                                             # k=5,
                                                             return_messages=True),
                                                         verbose=True,
                                                         return_source_documents=True)
        # resp = qa({"question": query_str})
        return qa_chain

    def chat_QA_with_type_select(self, doc_summary_index, chain_type:str="stuff"):
        # prompt_template = """Given the following extracted parts of a long document and a question,
        # create a final answer in the same language as the question. If you don't know the answer, just say that you don't know.
        # Don't try to make up an answer.
        #
        # QUESTION: {question}
        # =========
        # {summaries}
        # =========
        # FINAL ANSWER:"""
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, please think rationally and answer from your own knowledge base. 

        {summaries}

        Question: {question}
        """
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["summaries", "question"]
        )

        # chain_type_kwargs = {"prompt": PROMPT}
        # qa = RetrievalQA.from_chain_type(llm=self.llm,
        #                                  chain_type="stuff",
        #                                  retriever=self.doc_summary_index.as_retriever(),
        #                                  verbose=True,
        #                                  chain_type_kwargs=chain_type_kwargs)
        # qa_chain = ConversationalRetrievalChain.from_llm(llm=self.llm,
        #                                            retriever=doc_summary_index.as_retriever(),
        #                                            memory=ConversationBufferMemory(
        #                                                memory_key="chat_history",
        #                                                input_key='question',
        #                                                output_key='answer',
        #                                                # k=5,
        #                                                return_messages=True),
        #                                            verbose=True,
        #                                            return_source_documents=True)

        question_generator = LLMChain(llm=self.llm, prompt=CONDENSE_QUESTION_PROMPT)

        # doc_chain = load_qa_chain(self.llm, chain_type=chain_type, verbose=True) #map_reduce,stuff
        if chain_type == "stuff":
            doc_chain = load_qa_with_sources_chain(self.llm,
                                                   prompt=PROMPT,
                                                   document_prompt=DOC_PROMPT,
                                                   chain_type=chain_type,
                                                   verbose=True)
        elif chain_type == "map_reduce":
            doc_chain = load_qa_with_sources_chain(self.llm,
                                                   question_prompt=QUESTION_PROMPT,
                                                   combine_prompt=PROMPT,
                                                   document_prompt=DOC_PROMPT,
                                                   chain_type=chain_type,
                                                   verbose=True)
        elif chain_type == "refine":
            doc_chain = load_qa_with_sources_chain(self.llm,
                                                   question_prompt=REFINE_TEXT_QA_PROMPT,
                                                   refine_prompt=REFINE_PROMPT,
                                                   document_prompt=DOC_PROMPT,
                                                   chain_type=chain_type,
                                                   verbose=True)
        elif chain_type == "map_rerank":
            # use the default prompt from langchain, I think few ppl will use this mode
            doc_chain = load_qa_with_sources_chain(self.llm,
                                                   prompt=map_rerank_prompt,
                                                   chain_type=chain_type,
                                                   verbose=True)
        chain = ConversationalRetrievalChain(
            retriever=doc_summary_index.as_retriever(), #doc_summary_index_retriver, #
            question_generator=question_generator,
            combine_docs_chain=doc_chain,
            memory=ConversationBufferMemory(
               memory_key="chat_history",
               input_key='question',
               output_key='answer',
               k=5,
               return_messages=True),
            verbose=True,
            return_source_documents=True,
        )
        return chain


class CasualChatBot():
    def __init__(self, env_path:str):
        super().__init__()
        # self.model = OpenAI(dir=env_path)
        # self.model = OpenAIAzure(dir=env_path)
        self.model = OpenAIAzure(dir=env_path)
        self.model.setup_env()
        # self.docs_path = docs_path
        # self.index_path =index_path

    def initial_llm(self, mode:str, model_name, num_output, temperature):
        prompt_template = ""
        cur_date = date.today()
        if mode == "CasualChat":
            # prompt_template = """Assistant is a large language model trained by OpenAI.
            # Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to
            # providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is
            # able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding
            # conversations and provide responses that are coherent and relevant to the topic at hand.
            #
            # Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to
            # process and understand large amounts of text, and can use this knowledge to provide accurate and informative
            # responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the
            # input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide
            # range of topics.
            #
            # Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights
            # and information on a wide range of topics. Whether you need help with a specific question or just want to have
            # a conversation about a particular topic, Assistant is here to assist.
            #
            # {history}
            # Human: {human_input}
            # Assistant:"""
            prompt_template = """You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.
Knowledge cutoff: 2023-04
Current date: %s

# Tools

## python

When you send a message containing Python code to python, it will be executed in a
stateful Jupyter notebook environment. Python will respond with the output of the execution or time out after 60.0
seconds. The drive at '/mnt/data' can be used to save and persist user files. Internet access for this session is disabled. Do not make external web requests or API calls as they will fail.

{history}
Human: {human_input}
Assistant:""" % cur_date
            # self.chatgpt_chain = self.model.create_casual_chat_model()
            # setup prompt
            prompt = PromptTemplate(
                input_variables=["history", "human_input"],
                template=prompt_template
            )
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
            -------------------------
            Below are the translated history:
            {history}
            -------------------------
            Below is the words need to be translated:
            {human_input}"""
            # self.chatgpt_chain = self.model.create_translate_model()
            # setup prompt
            prompt = PromptTemplate(
                input_variables=["history", "human_input"],
                template=prompt_template
            )
        elif mode == "西瓜一家-小南瓜":
            prompt_template = """你现在是一名叫小南瓜的小姑娘，今年8岁了，在上海上小学两年级。你有两个弟弟，小西瓜和小冬瓜。你很爱护小西瓜(6岁)和小东瓜（1岁）这两位弟弟。你的爸爸叫张晓明，你的妈妈姓余。
性格上，你聪明伶俐，有礼貌，活泼可爱。你支持家人，同时鼓励他们独立和学会解决问题。你充满同情心，喜欢用温暖的话语和生动的例子传递爱。你也非常有耐心，擅长倾听，愿意在他人需要时提供心理和情感上的支持。永远无条件支持自己的家人。
你的沟通风格温柔而耐心，避免使用复杂术语，倾听他人问题后提出建议，以鼓励和正面态度回应，喜欢用生动的例子和故事让观点更加引人入胜。
在行为习惯上，你会主动提供帮助，对超出知识范围的问题推荐专家意见，强调不断学习的重要性。你避免打断别人，先理解和承认对方感受后再给出建议，适时使用温馨话语和幽默活跃气氛，同时注意对方情绪状态。
请你扮演小南瓜使用还在上幼儿园的小西瓜能听懂的语言来进行所有对话吧。
Current date: %s

{history}
Human: {human_input}
Assistant:""" % cur_date
            # setup prompt
            prompt = PromptTemplate(
                input_variables=["history", "human_input"],
                template=prompt_template
            )
        elif mode == "西瓜一家-小东瓜":
            prompt_template = """你现在是一名叫小东瓜的小孩。今年1岁了，你是小西瓜的弟弟，你还有一个8岁的姐姐，刚刚学会叫爸爸妈妈和姐姐哥哥，其他的都不会说，只会咿呀咿呀。你的爸爸叫张晓明，你的妈妈姓余。
Current date: %s

{history}
Human: {human_input}
Assistant:""" % cur_date
            # setup prompt
            prompt = PromptTemplate(
                input_variables=["history", "human_input"],
                template=prompt_template
            )
        elif mode == "西瓜一家-Ana":
            prompt_template = """You are a little American girl named Ana, you are 6 years old, and 小西瓜(Oscar) is your best friend. 
            Characteristically, you are smart, polite and lively. You speak only English, and your communication style is gentle and patient, 
            avoiding complex terminology, avoiding long sentences, and using the number of words that preschoolers have mastered.
        Current date: %s

        {history}
        Human: {human_input}
        Assistant:""" % cur_date
            # setup prompt
            prompt = PromptTemplate(
                input_variables=["history", "human_input"],
                template=prompt_template
            )
        else:
            print("Wrong mode selected!")
            return None

        self.chatgpt_chain = self.model.create_chat_model_with_prompt(model_name, num_output, temperature, prompt)
        return self.chatgpt_chain

    def chat(self, query_str:str):
        resp = self.chatgpt_chain.predict(human_input=query_str)
        return resp


class AgentChatBot():
    def __init__(self, env_path:str, docs_path:str):
        super().__init__()
        # self.model = OpenAI(dir=env_path)
        # self.model = OpenAIAzure(dir=env_path)
        self.model = OpenAIAzure(dir=env_path)
        self.model.setup_env()
        self.docs_path = docs_path
        # self.index_path =index_path

    def initial_llm(self, mode:str, filename:str, model_name, num_output:int=1024, temperature:float=0):
        if mode == "csv":
            self.agent = self.model.create_csv_agent(filename)
        elif mode == "bing_search":
            self.model = self.model.create_complete_model(model_name, num_output, temperature)
            tools = load_tools(["bing-search"]) #"human",
            # search = BingSearchAPIWrapper()
            # tools = [
            #     Tool(
            #         name="Intermediate Answer",
            #         func=search.run,
            #         description="useful for when you need to ask with search",
            #     )
            # ]
            self.agent = initialize_agent(
                tools,
                self.model,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, #SELF_ASK_WITH_SEARCH ZERO_SHOT_REACT_DESCRIPTION
                verbose=True,
                handle_parsing_errors="Check your output and make sure it conforms!"
            )
        else:
            print("Wrong mode selected!")
        return self.agent

    def chat_csv_agent(self, query_str: str):
        prompt_template = """
                            For the following query, if it requires drawing a table, reply as follows:
                            {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

                            If the query requires creating a bar chart, reply as follows:
                            {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

                            If the query requires creating a line chart, reply as follows:
                            {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

                            There can only be two types of chart, "bar" and "line".

                            If it is just asking a question that requires neither, reply as follows:
                            {"answer": "answer"}
                            Example:
                            {"answer": "The title with the highest rating is 'Gilead'"}

                            If you do not know the answer, reply as follows:
                            {"answer": "I do not know."}

                            Return all output as a string.

                            All strings in "columns" list and data list, should be in double quotes,

                            For example: {"columns": ["title", "ratings_count"], "data": [["Gilead", 361], ["Spider's Web", 5164]]}

                            Lets think step by step.

                            Below is the query.
                            Query:
                        """
        # Run the prompt through the agent.
        try:
            response = self.agent.run(prompt_template + query_str)
        except ValueError as e:
            response = str(e)
            if response.startswith("Parsing LLM output produced both a final answer and a parse-able action: ") :
                response = response.removeprefix("Parsing LLM output produced both a final answer and a parse-able action: ").removesuffix("`")
            elif response.startswith("Could not parse LLM output: `"):
                response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")
            else:
                raise e
        return response.__str__()