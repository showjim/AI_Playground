import os
import shutil
from dotenv import load_dotenv
import time
from llama_index import (
    GPTListIndex,
    SimpleDirectoryReader,
    PromptHelper,
    LLMPredictor,
    ServiceContext,
    ResponseSynthesizer,
    StorageContext
)
from llama_index.indices.document_summary import GPTDocumentSummaryIndex
from langchain.chat_models import ChatOpenAI

from llama_index.indices.loading import load_index_from_storage

class OpenAI():
    def __int__(self):
        super().__init__()
        self.WORK_ENV_DIR = './'
        self.ENV_FILE = 'key.txt'
        self.config_details = {}
    def setup_env(self):
        # Load OpenAI key
        if os.path.exists(os.path.join(self.WORK_ENV_DIR, self.ENV_FILE)):
            shutil.copyfile(os.path.join(self.WORK_ENV_DIR, self.ENV_FILE), ".env")
            load_dotenv()
        else:
            raise APIKeyNotFoundError("key.txt with OpenAI API is required")

    def create_chat_model(self):
        # # LLM Predictor (gpt-4)
        # define prompt helper
        # set maximum input size
        max_input_size = 4096
        # set number of output tokens
        num_output = 512  # 256
        # set maximum chunk overlap
        max_chunk_overlap = 20
        prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

        llm_predictor_chatgpt = LLMPredictor(llm=ChatOpenAI(temperature=0.2, model_name="gpt-4", max_tokens=num_output))
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor_chatgpt, prompt_helper=prompt_helper,
                                                       chunk_size_limit=1024)
        return service_context

    def load_docs(self, path:str):
        # load documents
        if os.path.isdir(path):
            documents = SimpleDirectoryReader(path).load_data()
        else:
            raise DirectoryIsNotGivenError("Directory is required to load documents")
        return documents

    def build_index(self, service_context, documents, path):
        # default mode of building the index
        response_synthesizer = ResponseSynthesizer.from_args(response_mode="tree_summarize", use_async=True)
        doc_summary_index = GPTDocumentSummaryIndex.from_documents(
            documents,
            service_context=service_context,
            response_synthesizer=response_synthesizer
        )
        doc_summary_index.storage_context.persist(path)

    def rebuild_index_from_dir(self, path):
        # rebuild storage context
        storage_context = StorageContext.from_defaults(persist_dir=path)
        doc_summary_index = load_index_from_storage(storage_context)
        return doc_summary_index

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