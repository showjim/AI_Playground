import os, json
import shutil
import openai
from dotenv import load_dotenv
from llama_index import (
    GPTListIndex,
    SimpleDirectoryReader,
    PromptHelper,
    LLMPredictor,
    ServiceContext,
    ResponseSynthesizer,
    StorageContext,
    LangchainEmbedding
)
from llama_index.indices.document_summary import GPTDocumentSummaryIndex
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from llama_index.indices.loading import load_index_from_storage
from langchain.llms import AzureOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import HumanMessage

class OpenAI():
    def __init__(self, dir="./", env='key.txt'):
        super().__init__()
        self.WORK_ENV_DIR = dir
        self.ENV_FILE = env

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

    def rebuild_index_from_dir(self, path, service_context):
        # rebuild storage context
        storage_context = StorageContext.from_defaults(persist_dir=path)
        doc_summary_index = load_index_from_storage(storage_context,service_context=service_context)
        return doc_summary_index

class OpenAIAzure():
    def __init__(self, dir="./", env='key.txt'):
        super().__init__()

        self.WORK_ENV_DIR = dir
        self.ENV_FILE = env
        self.config_details = {}
    def setup_env(self):
        # Load OpenAI key
        if os.path.exists(os.path.join(self.WORK_ENV_DIR, self.ENV_FILE)):
            shutil.copyfile(os.path.join(self.WORK_ENV_DIR, self.ENV_FILE), ".env")
            load_dotenv()
        else:
            raise APIKeyNotFoundError("key.txt with OpenAI API is required")

        # Load config values
        if os.path.exists(os.path.join(r'config.json')):
            with open(r'config.json') as config_file:
                self.config_details = json.load(config_file)

            # Setting up the embedding model
            self.embedding_model_name = self.config_details['EMBEDDING_MODEL']
            openai.api_type = "azure"
            openai.api_base = self.config_details['OPENAI_API_BASE']
            openai.api_version = self.config_details['OPENAI_API_VERSION']
            openai.api_key = os.getenv("OPENAI_API_KEY")
        else:
            raise AzureConfigNotFoundError("config.json with Azure OpenAI config is required")

    def create_chat_model(self):
        # max LLM token input size
        max_input_size = 3900  # 4096
        # set number of output tokens
        num_output = 1024  # 512
        # set maximum chunk overlap
        max_chunk_overlap = 20
        llm = AzureChatOpenAI(deployment_name=self.config_details['CHATGPT_MODEL'],
                               openai_api_key=openai.api_key,
                               openai_api_base=openai.api_base,
                               openai_api_type=openai.api_type,
                               openai_api_version=self.config_details['OPENAI_API_VERSION'],
                               max_tokens=num_output,
                               temperature=0.2,
                              # model_kwargs={'engine': self.config_details['CHATGPT_MODEL']},
                               )
        test =llm([HumanMessage(content="Translate this sentence from English to French. I love programming.")])
        # llm = AzureOpenAI(deployment_name=self.config_details['CHATGPT_MODEL'],
        #                   max_tokens=num_output,
        #                   temperature=0.2,
        #                   # model_kwargs={'engine': self.config_details['CHATGPT_MODEL']},
        #                   model_kwargs={
        #                       "api_key": openai.api_key,
        #                       "api_base": openai.api_base,
        #                       "api_type": openai.api_type,
        #                       "api_version": self.config_details['OPENAI_API_VERSION'],
        #                       "engine": self.config_details['CHATGPT_MODEL'],
        #                   }
        #                   )
        llm_predictor = LLMPredictor(llm=llm)

        # You need to deploy your own embedding model as well as your own chat completion model
        embedding_llm = LangchainEmbedding(
            OpenAIEmbeddings(
                model=self.embedding_model_name,
                deployment=self.embedding_model_name,
                openai_api_key=openai.api_key,
                openai_api_base=openai.api_base,
                openai_api_type=openai.api_type,
                openai_api_version=self.config_details['EMBEDDING_MODEL_VERSION'],
                # model_kwargs={'engine': self.config_details['CHATGPT_MODEL']}
            ),
            embed_batch_size=1,
        )
        prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
        service_context = ServiceContext.from_defaults(
            llm_predictor=llm_predictor,
            embed_model=embedding_llm,
            prompt_helper=prompt_helper,
            chunk_size_limit=1024
        )

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

    def rebuild_index_from_dir(self, path, service_context):
        # rebuild storage context
        storage_context = StorageContext.from_defaults(persist_dir=path)
        doc_summary_index = load_index_from_storage(storage_context,service_context=service_context)
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

class AzureConfigNotFoundError(Exception):
    """
    Raised when the API key is not defined/declared.

    Args:
        Exception (Exception): APIKeyNotFoundError
    """