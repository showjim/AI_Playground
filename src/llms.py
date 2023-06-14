import glob
import os, json
import shutil
from multiprocessing import Pool
from typing import List

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

from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import HumanMessage

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    DirectoryLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import PromptTemplate
# The vectorstore we'll be using
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

# The LangChain component we'll use to get the documents
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from tqdm import tqdm

from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.llms import AzureOpenAI



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


class OpenAIAzureLangChain():
    def __init__(self, dir="./", env='key.txt'):
        super().__init__()

        self.WORK_ENV_DIR = dir
        self.ENV_FILE = env
        self.config_details = {}
        # Map file extensions to document loaders and their arguments
        self.LOADER_MAPPING = {
            ".csv": (CSVLoader, {}),
            # ".docx": (Docx2txtLoader, {}),
            ".doc": (UnstructuredWordDocumentLoader, {}),
            ".docx": (UnstructuredWordDocumentLoader, {}),
            ".enex": (EverNoteLoader, {}),
            ".epub": (UnstructuredEPubLoader, {}),
            ".html": (UnstructuredHTMLLoader, {}),
            ".md": (UnstructuredMarkdownLoader, {}),
            ".odt": (UnstructuredODTLoader, {}),
            ".pdf": (PDFMinerLoader, {}),
            ".ppt": (UnstructuredPowerPointLoader, {}),
            ".pptx": (UnstructuredPowerPointLoader, {}),
            ".txt": (TextLoader, {"encoding": "utf8"}),
            # Add more mappings for other file extensions and loaders as needed
        }

    def setup_env(self):
        # Load OpenAI key
        if os.path.exists(os.path.join(self.WORK_ENV_DIR, self.ENV_FILE)):
            shutil.copyfile(os.path.join(self.WORK_ENV_DIR, self.ENV_FILE), ".env")
            load_dotenv()
        else:
            raise APIKeyNotFoundError("key.txt with OpenAI API is required")

        # Load config values
        if os.path.exists(os.path.join(self.WORK_ENV_DIR, r'config.json')):
            with open(os.path.join(self.WORK_ENV_DIR, r'config.json')) as config_file:
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
                               )


        # You need to deploy your own embedding model as well as your own chat completion model
        embeddings = OpenAIEmbeddings(deployment=self.config_details['EMBEDDING_MODEL'], chunk_size=1)


        return llm, embeddings

    def get_all_files(self, source_dir):
        all_files = []
        for ext in self.LOADER_MAPPING:
            all_files.extend(
                glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
            )
        return all_files

    def load_single_document(self, file_path):
        ext = "." + file_path.rsplit(".", 1)[-1]
        if ext in self.LOADER_MAPPING:
            loader_class, loader_args = self.LOADER_MAPPING[ext]
            loader = loader_class(file_path, **loader_args)
            return loader.load()[0]

        raise ValueError(f"Unsupported file extension '{ext}'")

    def load_documents(self, all_files: list) -> List[Document]:
        """
        Loads all documents from the source documents directory
        """
        # with Pool(processes=os.cpu_count()) as pool:
        documents = []
        with tqdm(total=len(all_files), desc='Loading new documents', ncols=80) as pbar:
            for doc in all_files:
                documents.append(self.load_single_document(doc))
                pbar.update()

        return documents
    def load_docs(self, path:str):
        # load documents
        if os.path.isdir(path):
            # documents = SimpleDirectoryReader(path).load_data()
            print(f"Loading documents from {path}")
            all_files = self.get_all_files(path)
            documents = self.load_documents(all_files)
        else:
            raise DirectoryIsNotGivenError("Directory is required to load documents")
        return documents

    def build_index(self, embeddings, documents, path):
        print(f"Loaded {len(documents)} new documents")
        chunk_size = 2048
        chunk_overlap = 100
        # Get your splitter ready
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # Split your docs into texts
        texts = text_splitter.split_documents(documents)
        print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")

        docsearch = FAISS.from_documents(texts, embeddings)
        docsearch.save_local(path)

    def rebuild_index_from_dir(self, path, embeddings):
        # rebuild storage context
        doc_summary_index = FAISS.load_local(path, embeddings)
        return doc_summary_index

    def create_casual_chat_model(self):
        # setup prompt
        template = """Assistant is a large language model trained by OpenAI.
        
        Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to 
        providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is 
        able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding 
        conversations and provide responses that are coherent and relevant to the topic at hand.

        Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to 
        process and understand large amounts of text, and can use this knowledge to provide accurate and informative 
        responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the 
        input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide 
        range of topics.

        Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights 
        and information on a wide range of topics. Whether you need help with a specific question or just want to have 
        a conversation about a particular topic, Assistant is here to assist.

        {history}
        Human: {human_input}
        Assistant:"""

        prompt = PromptTemplate(
            input_variables=["history", "human_input"],
            template=template
        )


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
                               temperature=1.0,
                               )
        # llm = AzureOpenAI(deployment_name=self.config_details['CHATGPT_MODEL'],
        #                   model_name=self.config_details['CHATGPT_MODEL'],
        #                   # openai_api_key=openai.api_key,
        #                   # openai_api_base=openai.api_base,
        #                   # openai_api_type=openai.api_type,
        #                   # openai_api_version=self.config_details['OPENAI_API_VERSION'],
        #                   # max_tokens=num_output,
        #                   # temperature=0.2,
        #                   )
        chatgpt_chain = LLMChain(
            llm=llm, #OpenAI(temperature=0),
            prompt=prompt,
            verbose=True,
            memory=ConversationBufferWindowMemory(k=5),
        )

        return chatgpt_chain

    def create_translate_model(self):
        # setup prompt
        template = """You are a professional translator. Translate anything that I say to Chinese or English in a natural manner. 
        Only return the translate result. Don't interpret it. Please use the same format of input in output answer.
        
        Below are the history of translation：
        ------------------------------
        {history}
        -----------------------------
        
        Below is the words need to be translated:
        Human: {human_input}
        Assistant:"""

        prompt = PromptTemplate(
            input_variables=["history", "human_input"],
            template=template
        )

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
                              temperature=1.0,
                              )
        # llm = AzureOpenAI(deployment_name=self.config_details['CHATGPT_MODEL'],
        #                   model_name=self.config_details['CHATGPT_MODEL'],
        #                   # openai_api_key=openai.api_key,
        #                   # openai_api_base=openai.api_base,
        #                   # openai_api_type=openai.api_type,
        #                   # openai_api_version=self.config_details['OPENAI_API_VERSION'],
        #                   # max_tokens=num_output,
        #                   # temperature=0.2,
        #                   )
        chatgpt_chain = LLMChain(
            llm=llm,  # OpenAI(temperature=0),
            prompt=prompt,
            verbose=True,
            memory=ConversationBufferWindowMemory(k=5),
        )

        return chatgpt_chain



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