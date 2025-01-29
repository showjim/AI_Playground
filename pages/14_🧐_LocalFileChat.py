# Adapted from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps#build-a-simple-chatbot-gui-with-streaming
import os, base64, gc, uuid, re
from typing import List

import openai, glob
from pathlib import Path

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from llama_index.embeddings.ollama import OllamaEmbedding #HuggingFaceEmbedding
from llama_index.core import (
    VectorStoreIndex, ServiceContext,
    SimpleDirectoryReader, StorageContext,
    load_index_from_storage)
from llama_index.core.vector_stores.simple import (
    DEFAULT_VECTOR_STORE,
    NAMESPACE_SEP,
)
from llama_index.vector_stores.faiss import FaissVectorStore

import streamlit as st
import faiss



# dimensions of nomic-embed-text
d = 768 #1536
faiss_index = faiss.IndexFlatL2(d)
BASE_URL = 'http://127.0.0.1:11434/'


@st.cache_resource
def load_llm(model: str = "deepseek-r1:1.5b", temperature: float = 0.2):
    global BASE_URL
    llm = Ollama(model=model, request_timeout=120.0, temperature=temperature,base_url=BASE_URL)
    return llm


def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()


def display_pdf(file):
    # Opening file from file path

    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")

    # Embedding PDF in HTML
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                        style="height:100vh; width:100%"
                    >
                    </iframe>"""

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)


def set_reload_flag():
    # st.write("New document need upload")
    st.session_state["LocalFileReloadFlag"] = True


def set_reload_db_flag():
    st.session_state["IndexReloadFlag"] = True


def get_all_files_list(source_dir, ext:str = "faiss"):
    all_files = []
    result = []
    all_files.extend(
        glob.glob(os.path.join(source_dir, f"*.{ext}"), recursive=False)
    )
    for filepath in all_files:
        file_name = Path(filepath).name
        if file_name.startswith("default_"):
            result.append(file_name)
    return result

@st.cache_resource
def load_single_vectordb(workDir:str, vsFileName:str):
    vsFilePath = os.path.join(workDir, f"{vsFileName}")
    # load index from disk
    vector_store = FaissVectorStore.from_persist_path(persist_path=vsFilePath)  # .from_persist_dir(workDir)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir=workDir
    )
    index = load_index_from_storage(storage_context=storage_context)
    return index

@st.cache_resource
def load_vectordbs(workDir:str, all_files: List[str]):
    nodes = []
    for i in range(len(all_files)):
        filename = all_files[i]
        index = load_single_vectordb(workDir, filename)
        # merge index with nodes
        nodes_dict = index.storage_context.docstore.docs
        nodes = list(nodes_dict.values())
        # for doc_id, node in nodes_dict.items():
            # necessary to avoid re-calc of embeddings
            # node.embedding = embedding_dict[doc_id]
            # nodes.append(node)
    final_index = VectorStoreIndex(nodes=nodes)
    return final_index


def create_query_engine(index):
    # Create the query engine, where we use a cohere reranker on the fetched nodes
    query_engine = index.as_query_engine(streaming=True) #, similarity_top_k=3, verbose=True,) #response_mode="refine", "compact"(default), "tree_summarize"

    # ====== Customise prompt template ======
    qa_prompt_tmpl_str = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information above I want you to think step by step to answer the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
    )
    return query_engine

def main():
    if "id" not in st.session_state:
        st.session_state.id = uuid.uuid4()
        st.session_state.file_cache = {}
    if "OpenChatReloadFlag" not in st.session_state:
        st.session_state["OpenChatReloadFlag"] = True
    if "QueryEngine" not in st.session_state:
        st.session_state["QueryEngine"] = None
    if "IndexReloadFlag" not in st.session_state:
        st.session_state["IndexReloadFlag"] = True
    if "LocalVectorDB" not in st.session_state:
        st.session_state["LocalVectorDB"] = None

    session_id = st.session_state.id
    work_path = os.path.abspath('.')
    workDir = os.path.join(work_path, "workDir")

    with st.sidebar:
        st.header(f"RAG Setting")
        # select Ollama base url
        option = st.selectbox(
            "Select Base URL",
            ("http://localhost:11434/", "http://ollama:11434/", "http://127.0.0.1:11434/", "Another option..."),
        )
        global BASE_URL
        # Create text input for user entry
        if option == "Another option...":
            BASE_URL = st.text_input("Enter your other option...")
        else:
            BASE_URL = option

        client = openai.OpenAI(
            base_url=BASE_URL + 'v1/',

            # required but ignored
            api_key='ollama',
        )

        list_completion = client.models.list()
        models = [model.id for model in list_completion.data]
        model = st.selectbox(
            label="Select Model",
            options=models,
            index=0,
            on_change=set_reload_db_flag)

        # setup llm & embedding model
        llm = load_llm(model)
        Settings.llm = llm
        embed_model = OllamaEmbedding(model_name="nomic-embed-text:latest", base_url=BASE_URL)
        # Creating an index over loaded data
        Settings.embed_model = embed_model

        st.header(f"Add your documents!")
        uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")
        if st.button("Upload"):
            if uploaded_file:
                try:
                    # with tempfile.TemporaryDirectory() as temp_dir:
                    file_path = os.path.join(workDir, uploaded_file.name)

                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    file_key = f"{session_id}-{uploaded_file.name}"
                    st.write("Indexing your document...")

                    if os.path.exists(workDir):
                        loader = SimpleDirectoryReader(
                            # input_dir=temp_dir,
                            input_files=[file_path],
                            required_exts=[".pdf"],
                            recursive=False
                        )
                    else:
                        st.error('Could not find the file you uploaded, please check again...')
                        st.stop()

                    docs = loader.load_data()

                    # Creating an index over loaded data
                    vector_store = FaissVectorStore(faiss_index=faiss_index)
                    storage_context = StorageContext.from_defaults(vector_store=vector_store)
                    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context, show_progress=True)

                    # Store the index in local index file
                    ext = os.path.splitext(uploaded_file.name)
                    vsFileName = f"{ext[0]}.faiss"
                    vsFilePath = os.path.join(workDir, f"{DEFAULT_VECTOR_STORE}{NAMESPACE_SEP}{vsFileName}")
                    index.storage_context.persist(persist_dir=workDir, vector_store_fname=vsFileName)

                    # Inform the user that the file is processed and Display the PDF uploaded
                    st.success("PDF to Vector DB Done!")
                    # display_pdf(uploaded_file)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.stop()
        # select the specified index base(s)
        index_file_list = get_all_files_list(workDir, "faiss")
        options = st.multiselect('2.What file do you want to exam?',
                                 index_file_list,
                                 on_change=set_reload_db_flag)
        if len(options) > 0:
            if st.session_state["IndexReloadFlag"] == True:
                with st.spinner('Load Index DB'):
                    index = load_vectordbs(workDir, options)
                    query_engine = create_query_engine(index)
                    st.session_state["QueryEngine"] = query_engine
                    st.session_state["IndexReloadFlag"] = False
            if (st.session_state["QueryEngine"] is not None):
                st.write("✅ " + ", ".join(options) + " Index DB Loaded")
        else:
            st.session_state["QueryEngine"] = None


    col1, col2 = st.columns([6, 1])

    # st.markdown("""
    #     # Agentic RAG powered by <img src="data:image/png;base64,{}" width="120" style="vertical-align: -3px;">
    # """.format(base64.b64encode(open("assets/crewai.png", "rb").read()).decode()), unsafe_allow_html=True)


    with col1:
        # st.header(f"🧐 Chat with Docs using Local LLMs")
        st.markdown("""
        ## 🧐 Local File Chat
        ##### RAG powered by <img src="data:image/png;base64,{}" width="25" style="vertical-align: -5px;">
    """.format(base64.b64encode(open("img/logo/ollama-logo.png", "rb").read()).decode()), unsafe_allow_html=True)

    with col2:
        st.button("Clear ↺", on_click=reset_chat)

    # Initialize chat history
    if "messages" not in st.session_state:
        reset_chat()

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            with st.chat_message(message["role"]):
                with st.expander("See thinking"):
                    st.markdown(message["content"]["thinking"])
                st.markdown(message["content"]["answers"])
        else:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What's up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with (st.chat_message("assistant")):
            thinking_placeholder = st.empty()
            message_placeholder = st.empty()
            full_response = ""
            with st.spinner("preparing answer"):
                # Simulate stream of response with milliseconds delay
                streaming_response = st.session_state["QueryEngine"].query(prompt)

                for chunk in streaming_response.response_gen:
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")

                resp_thinking = ""
                resp_answer = ""
                if '<think>' in full_response and '</think>' in full_response:
                    split_resp = re.split('<think>|</think>', full_response)
                    resp_thinking = split_resp[1]
                    resp_answer = split_resp[2]
                    with thinking_placeholder.expander("See thinking"):
                        st.markdown(resp_thinking)
                message_placeholder.markdown(resp_answer)
                # st.session_state.context = ctx

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": {"answers": resp_answer, "thinking": resp_thinking}})

if __name__ == "__main__":
    main()