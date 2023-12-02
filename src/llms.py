import glob
import os, json
import shutil
from multiprocessing import Pool
from typing import List
from pathlib import Path
import pandas as pd
import openai
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI, AzureChatOpenAI

from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.schema import HumanMessage

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    PyMuPDFLoader,
    PyPDFLoader,
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
from langchain.prompts import PromptTemplate
# The vectorstore we'll be using
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

# The LangChain component we'll use to get the documents
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from tqdm import tqdm

from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.llms import AzureOpenAI
# from langchain.agents import create_pandas_dataframe_agent
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.callbacks.base import BaseCallbackHandler


# class StreamHandler(BaseCallbackHandler):
#     def __init__(self, container, initial_text=""):
#         self.container = container
#         self.text=initial_text
#     def on_llm_new_token(self, token: str, **kwargs) -> None:
#         # "/" is a marker to show difference
#         # you don't need it
#         self.text+=token+"/"
#         self.container.markdown(self.text)


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


class OpenAIAzure():
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
            ".pdf": (PyMuPDFLoader, {}), #PyMuPDFLoader PDFMinerLoader PyPDFLoader
            ".ppt": (UnstructuredPowerPointLoader, {}),
            ".pptx": (UnstructuredPowerPointLoader, {}),
            ".txt": (TextLoader,{}),# {"encoding": "utf8"}),
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

            #Text2Speech
            os.environ["SPEECH_KEY"] = os.getenv("SPEECH_KEY")
            os.environ["SPEECH_REGION"] = self.config_details['SPEECH_REGION']
        else:
            raise AzureConfigNotFoundError("config.json with Azure OpenAI config is required")

    def create_chat_model(self, model_name, num_output, temperature):
        # max LLM token input size
        # max_input_size = 3900  # 4096
        # set number of output tokens
        # num_output = 1024  # 512
        # set maximum chunk overlap
        # max_chunk_overlap = 20
        llm = AzureChatOpenAI(deployment_name=model_name, #self.config_details['CHATGPT_MODEL'],
                               openai_api_key=openai.api_key,
                               azure_endpoint=openai.api_base,
                               openai_api_type=openai.api_type,
                               openai_api_version=self.config_details['OPENAI_API_VERSION'],
                               max_tokens=num_output,
                               temperature=temperature,#0.2,
                               )


        # You need to deploy your own embedding model as well as your own chat completion model
        embeddings = AzureOpenAIEmbeddings(deployment=self.config_details['EMBEDDING_MODEL'],
                                      azure_endpoint=openai.api_base,
                                      openai_api_type=openai.api_type,
                                      # chunk_size=1,
        )


        return llm, embeddings

    def get_all_files(self, source_dir):
        all_files = []
        for ext in self.LOADER_MAPPING:
            all_files.extend(
                glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
            )
        return all_files

    def get_all_files_by_ext(self, source_dir, ext):
        all_files = []
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
        return all_files

    def load_single_document(self, file_path):
        ext = "." + file_path.rsplit(".", 1)[-1]
        if ext in self.LOADER_MAPPING:
            loader_class, loader_args = self.LOADER_MAPPING[ext]
            loader = loader_class(file_path, **loader_args)
            return loader.load() #[0]

        raise ValueError(f"Unsupported file extension '{ext}'")

    def load_documents(self, all_files: list) -> List[Document]:
        """
        Loads all documents from the source documents directory
        """
        # with Pool(processes=os.cpu_count()) as pool:
        documents = []
        with tqdm(total=len(all_files), desc='Loading new documents', ncols=80) as pbar:
            for doc in all_files:
                documents = documents + self.load_single_document(doc)
                pbar.update()

        return documents
    def load_docs(self, path:str):
        # load documents
        if os.path.isdir(path):
            # documents = SimpleDirectoryReader(path).load_data()
            print(f"Loading documents from {path}")
            all_files = self.get_all_files(path)
            documents = self.load_documents(all_files)
        elif os.path.isfile(path):
            print(f"Loading specific document from {path}")
            documents = self.load_documents([path])
        else:
            print(f"The path here is : {path}")
            raise DirectoryIsNotGivenError("Directory or file name is required to load documents")
        return documents

    def build_index(self, embeddings, documents, path, indexfilename):
        print(f"Loaded {len(documents)} new documents")
        chunk_size = 1000 #1000 #1024 #2048
        chunk_overlap = 100
        # Get your splitter ready
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # Split your docs into texts
        texts = text_splitter.split_documents(documents)
        print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
        if len(texts) == 0:
            print("Error: there is no texts found in the document!")
        else:
            docsearch = FAISS.from_documents(texts, embeddings)
            docsearch.save_local(path, indexfilename)

    def rebuild_index_from_dir(self, path, embeddings, index_name:str = ""):
        # rebuild storage context from directory or specified file name
        if index_name == "":
            all_files = self.get_all_files_by_ext(path, "faiss")
            for i in range(len(all_files)):
                filename = all_files[i]
                tmpfile = Path(filename).stem
                if i == 0:
                    doc_summary_index = FAISS.load_local(path, embeddings, tmpfile)
                else:
                    doc_summary_index.merge_from(FAISS.load_local(path, embeddings, tmpfile))
        else:
            doc_summary_index = FAISS.load_local(path, embeddings, index_name)
        return doc_summary_index

    def rebuild_index_by_list(self, path, embeddings, all_files):
        # all_files = self.get_all_files_by_ext(path, "faiss")
        for i in range(len(all_files)):
            filename = all_files[i]
            tmpfile = filename #Path(filename).stem, has done in get_all_files_list
            if i == 0:
                doc_summary_index = FAISS.load_local(path, embeddings, tmpfile)
            else:
                doc_summary_index.merge_from(FAISS.load_local(path, embeddings, tmpfile))
        return doc_summary_index

    def create_chat_model_with_prompt(self, model_name, num_output, temperature, prompt):
        # # setup prompt
        # prompt = PromptTemplate(
        #     input_variables=["history", "human_input"],
        #     template=template
        # )

        llm = AzureChatOpenAI(deployment_name=model_name, #self.config_details['CHATGPT_MODEL'],
                              openai_api_key=openai.api_key,
                              azure_endpoint=openai.api_base,
                              openai_api_type=openai.api_type,
                              openai_api_version=self.config_details['OPENAI_API_VERSION'],
                              max_tokens=num_output,
                              temperature=temperature,  # 1.0,
                              streaming=True,
                              # callbacks=[StreamHandler]
                              )
        # llm = AzureOpenAI(deployment_name=model_name, #self.config_details['CHATGPT_MODEL'],
        #                   model_name=self.config_details['CHATGPT_MODEL'],
        #                   # openai_api_key=openai.api_key,
        #                   # azure_endpoint=openai.api_base,
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

    def create_csv_agent(self, filename:str):
        # setup prompt

        # max LLM token input size
        num_output = 1024  # 512
        llm = AzureChatOpenAI(deployment_name=self.config_details['CHATGPT_MODEL'],
                              openai_api_key=openai.api_key,
                              azure_endpoint=openai.api_base,
                              openai_api_type=openai.api_type,
                              openai_api_version=self.config_details['OPENAI_API_VERSION'],
                              max_tokens=num_output,
                              temperature=0.2,
                              )
        # llm = AzureOpenAI(deployment_name=self.config_details['CHATGPT_MODEL'],
        #                   model_name=self.config_details['CHATGPT_MODEL'],
        #                   openai_api_key=openai.api_key,
        #                   azure_endpoint=openai.api_base,
        #                   openai_api_type=openai.api_type,
        #                   openai_api_version=self.config_details['OPENAI_API_VERSION'],
        #                   max_tokens=num_output,
        #                   temperature=0.2,
        #                   )
        # Read the CSV file into a Pandas DataFrame.
        df = pd.read_csv(filename)
        csv_agent = create_pandas_dataframe_agent(llm=llm,
                                                  df=df,
                                                  verbose=True,
                                                  # prefix=prompt_template,
                                                  # input_variables=["query","table"],
                                                  )

        return csv_agent

    def create_complete_model(self, model_name, num_output:int=1024, temperature:float=0.2):
        llm = AzureChatOpenAI(deployment_name=model_name,
                              openai_api_key=openai.api_key,
                              azure_endpoint=openai.api_base,
                              openai_api_type=openai.api_type,
                              openai_api_version=self.config_details['OPENAI_API_VERSION'],
                              max_tokens=num_output,
                              temperature=temperature, #0.2,
                              streaming=True,
                              )
        # llm = AzureOpenAI(deployment_name=model_name,
        #                   model_name=model_name,
        #                   openai_api_key=openai.api_key,
        #                   azure_endpoint=openai.api_base,
        #                   openai_api_type=openai.api_type,
        #                   openai_api_version=self.config_details['OPENAI_API_VERSION'],
        #                   max_tokens=num_output,
        #                   temperature=0, #0.2,
        #                   streaming=True,
        #                   )

        return llm



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