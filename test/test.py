import os, json
import shutil
import openai
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI, AzureChatOpenAI

from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import HumanMessage
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import PromptTemplate
# The vectorstore we'll be using
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

# The LangChain component we'll use to get the documents
from langchain.chains import RetrievalQA

# Load OpenAI key
if os.path.exists(os.path.join("../", "key.txt")):
    shutil.copyfile(os.path.join("../", "key.txt"), "../.env")
    load_dotenv()
else:
    print("key.txt with OpenAI API is required")

# Load config values
if os.path.exists(os.path.join(r'../config.json')):
    with open(r'../config.json') as config_file:
        config_details = json.load(config_file)

    # Setting up the embedding model
    embedding_model_name = config_details['EMBEDDING_MODEL']
    openai.api_type = "azure"
    openai.api_base = config_details['OPENAI_API_BASE']
    openai.api_version = config_details['OPENAI_API_VERSION']
    openai.api_key = os.getenv("OPENAI_API_KEY")
else:
    print("config.json with Azure OpenAI config is required")

llm = AzureChatOpenAI(deployment_name=config_details['CHATGPT_MODEL'],
                               openai_api_key=openai.api_key,
                               openai_api_base=openai.api_base,
                               openai_api_type=openai.api_type,
                               openai_api_version=config_details['OPENAI_API_VERSION'],
                               max_tokens=512,
                               temperature=0.2,
                              # model_kwargs={'engine': self.config_details['CHATGPT_MODEL']},
                               )
embeddings = OpenAIEmbeddings(deployment=config_details['EMBEDDING_MODEL'], chunk_size=1)
# summary_chain = load_summarize_chain(src, chain_type="map_reduce")
# text = "This is a test document."
# query_result = embeddings.embed_query(text)
# print(query_result)
# Create our template
query = """
You are a helpful assistant to do meeting record.
Please summary this meeting record.
Please try to focus on the below requests, and use the bullet format to output the answers for each request: 
1. who attend the meeting?
2. Identify key decisions in the transcript.
3. What are the key action items in the meeting?
4. what are the next steps?
"""

# Create a LangChain prompt template that we can insert values to later
# prompt = PromptTemplate(
#     input_variables=["context", "question"],
#     template=template,
# )


loader = TextLoader(r'/tempDir/output/test.txt')
documents = loader.load()
# Get your splitter ready
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=100)

# Split your docs into texts
texts = text_splitter.split_documents(documents)

# There is a lot of complexity hidden in this one line. I encourage you to check out the video above for more detail
# chain = load_summarize_chain(src, chain_type="map_reduce", verbose=True)
# chain.run(texts)

docsearch = FAISS.from_documents(texts, embeddings)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())
a=qa.run(query)

print(a)