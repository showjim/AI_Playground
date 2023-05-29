import os.path
from pathlib import Path
from src.llms import OpenAI, OpenAIAzure, OpenAIAzureLangChain
from langchain.chains import RetrievalQA

class ChatBot():
    def __init__(self, docs_path:str, index_path:str, env_path:str):
        super().__init__()
        # self.model = OpenAI(dir=env_path)
        # self.model = OpenAIAzure(dir=env_path)
        self.model = OpenAIAzureLangChain(dir=env_path)
        self.model.setup_env()
        self.docs_path = docs_path
        self.index_path =index_path
        # self.service_context = self.model.create_chat_model()
        # self.documents = self.model.load_docs(docs_path)
        # self.model.build_index(self.service_context, self.documents, index_path)
        # self.doc_summary_index = self.model.rebuild_index_from_dir(index_path)
    def setup(self):
        DEFAULT_INDEX_FILE = "index.faiss"
        index_file = os.path.join(Path(self.index_path), Path(DEFAULT_INDEX_FILE))
        self.service_context = self.model.create_chat_model()
        if not os.path.exists(index_file):
            self.documents = self.model.load_docs(self.docs_path)
            self.model.build_index(self.service_context, self.documents, self.index_path)
        self.doc_summary_index = self.model.rebuild_index_from_dir(self.index_path, self.service_context)
    def chat(self, query_str:str):
        query_engine = self.doc_summary_index.as_query_engine(response_mode="tree_summarize")  # , streaming=True)
        response = query_engine.query(query_str)
        return response.response

    def setup_langchain(self):
        DEFAULT_INDEX_FILE = "index.faiss"
        index_file = os.path.join(Path(self.index_path), Path(DEFAULT_INDEX_FILE))
        self.llm, self.embedding = self.model.create_chat_model()
        if not os.path.exists(index_file):
            self.documents = self.model.load_docs(self.docs_path)
            self.model.build_index(self.embedding, self.documents, self.index_path)
        self.doc_summary_index = self.model.rebuild_index_from_dir(self.index_path, self.embedding)
    def chat_langchain(self, query_str:str):
        qa = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=self.doc_summary_index.as_retriever())
        resp = qa.run(query_str)
        return resp

class CasualChatBot():
    def __init__(self, env_path:str):
        super().__init__()
        # self.model = OpenAI(dir=env_path)
        # self.model = OpenAIAzure(dir=env_path)
        self.model = OpenAIAzureLangChain(dir=env_path)
        self.model.setup_env()
        # self.docs_path = docs_path
        # self.index_path =index_path

    def setup_langchain(self):
        self.chatgpt_chain = self.model.create_casual_chat_model()
        return self.chatgpt_chain

    def chat_langchain(self, query_str:str):
        resp = self.chatgpt_chain.predict(human_input=query_str)
        return resp