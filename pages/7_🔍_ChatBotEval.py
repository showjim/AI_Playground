import streamlit as st
import json, os, shutil, openai, csv
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

work_path = os.path.abspath('.')
def setup_env():
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

def set_reload_setting_flag():
    # st.write("New document need upload")
    st.session_state["evalreloadflag"] = True

def define_llm(model:str):
    llm = AzureChatOpenAI(deployment_name=model,
                          openai_api_key=openai.api_key,
                          openai_api_base=openai.api_base,
                          openai_api_type=openai.api_type,
                          openai_api_version=openai.api_version,
                          max_tokens=1024,
                          temperature=0.2,
                          # model_kwargs={'engine': self.config_details['CHATGPT_MODEL']},
                          )
    return llm

def define_retriver(retriver:str):
    pass

def define_splitter(splitter:str, chunk_size, chunk_overlap):
    if splitter == "RecursiveCharacterTextSplitter":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif splitter == "CharacterTextSplitter":
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter

def define_embedding():
    pass


def main():
    # Initial
    if "evalreloadflag" not in st.session_state:
        st.session_state["evalreloadflag"] = True

    # Setup Side Bar
    with st.sidebar:
        # 1. Model
        aa_llm_model = st.radio(label="LLM Model",
                                options=["gpt-35-turbo", "gpt-35-turbo-16k"],
                                index=0,
                                on_change=set_reload_setting_flag)
        # 2. Split
        aa_eval_q = st.slider(label="Number of eval questions",
                              min_value=1,
                              max_value=10,
                              value=5,
                              on_change=set_reload_setting_flag)
        aa_chunk_size = st.slider(label="Choose chunk size for splitting",
                                  min_value=500,
                                  max_value=2000,
                                  value=1000,
                                  on_change=set_reload_setting_flag)
        aa_overlap_size = st.slider(label="Choose overlap for splitting",
                                    min_value=0,
                                    max_value=100,
                                    value=200,
                                    on_change=set_reload_setting_flag)
        aa_split_methods = st.radio(label="Split method",
                                    options=["RecursiveTextSplitter", "CharacterTextSplitter"],
                                    index=0,
                                    on_change=set_reload_setting_flag)

        # 3. Retriver
        aa_retriver = st.radio(label="Choose retriever",
                               options=["Azure Cognitive Search", "OpenAI", "SVM"],
                               index=0,
                               on_change=set_reload_setting_flag)
        aa_chunk_num = st.select_slider("`Choose # chunks to retrieve`",
                                        options=[3, 4, 5, 6, 7, 8],
                                        on_change=set_reload_setting_flag)

        # 4. Embedding
        aa_embedding_method = st.radio(label="Choose embeddings",
                                       options=["Azure Cognitive Search", "OpenAI"],
                                       index=0,
                                       on_change=set_reload_setting_flag)

    if st.session_state["evalreloadflag"] == True:
        st.session_state["FileChat"].initial_llm(aa_llm_model, 2048, 0.2)
        st.session_state["evalreloadflag"] = False

    # Main
    st.header("`Demo auto-evaluator`")
    file_paths = st.file_uploader("1.Upload document files to generate QAs",
                                  type=["pdf", "txt", "pptx", "docx", "html"],
                                  accept_multiple_files=True)

    if st.button("Upload"):
        if file_paths is not None or len(file_paths) > 0:
            # save file
            with st.spinner('Reading file'):
                pass


if __name__ == "__main__":
    main()
