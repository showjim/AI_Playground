import os.path
from pathlib import Path
from src.llms import OpenAI, OpenAIAzure
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory

from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT

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

    def initial_llm(self):
        self.llm, self.embedding = self.model.create_chat_model()

    def setup_vectordb(self, filname:str):
        DEFAULT_INDEX_FILE = Path(filname).stem + ".faiss"
        index_file = os.path.join(Path(self.index_path), Path(DEFAULT_INDEX_FILE))
        # self.llm, self.embedding = self.model.create_chat_model()
        if not os.path.exists(index_file):
            self.documents = self.model.load_docs(filname) #(self.docs_path)
            self.model.build_index(self.embedding, self.documents, self.index_path, Path(filname).stem)
        self.doc_summary_index = self.model.rebuild_index_from_dir(self.index_path, self.embedding)
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
        If you don't know the answer, please think rationally and answer from your own knowledge base 

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

    def chat_QA_map_reduce(self, doc_summary_index):
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, please think rationally and answer from your own knowledge base 

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
        doc_chain = load_qa_chain(self.llm, chain_type="map_reduce") #map_reduce,stuff

        chain = ConversationalRetrievalChain(
            retriever=doc_summary_index.as_retriever(),
            question_generator=question_generator,
            combine_docs_chain=doc_chain,
            memory=ConversationBufferMemory(
               memory_key="chat_history",
               input_key='question',
               output_key='answer',
               # k=5,
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

    def initial_llm(self, mode:str):
        if mode == "CasualChat":
            self.chatgpt_chain = self.model.create_casual_chat_model()
        elif mode == "Translate":
            self.chatgpt_chain = self.model.create_translate_model()
        else:
            print("Wrong mode selected!")
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

    def initial_llm(self, mode:str, filename:str):
        if mode == "csv":
            self.agent = self.model.create_csv_agent(filename)
        else:
            print("Wrong mode selected!")
        return self.agent

    def chat_csv_agrent(self, query_str: str):
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
        response = self.agent.run(prompt_template + query_str)
        return response.__str__()