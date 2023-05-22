from llm.llms import OpenAI, OpenAIAzure

class ChatBot():
    def __init__(self, docs_path:str, index_path:str, env_path:str):
        super().__init__()
        # self.model = OpenAI(dir=env_path)
        self.model = OpenAIAzure(dir=env_path)
        self.model.setup_env()
        self.docs_path = docs_path
        self.index_path =index_path
        # self.service_context = self.model.create_chat_model()
        # self.documents = self.model.load_docs(docs_path)
        # self.model.build_index(self.service_context, self.documents, index_path)
        # self.doc_summary_index = self.model.rebuild_index_from_dir(index_path)
    def setup(self):
        self.service_context = self.model.create_chat_model()
        self.documents = self.model.load_docs(self.docs_path)
        self.model.build_index(self.service_context, self.documents, self.index_path)
        self.doc_summary_index = self.model.rebuild_index_from_dir(self.index_path, self.service_context)
    def chat(self, query_str:str):
        query_engine = self.doc_summary_index.as_query_engine(response_mode="tree_summarize")  # , streaming=True)
        response = query_engine.query(query_str)
        return response.response