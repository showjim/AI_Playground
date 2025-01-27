import time

import streamlit as st
import os
from pathlib import Path
from src.chat import ChatBot, StreamHandler
import nltk
import ssl
from langchain_community.retrievers import (
    AzureCognitiveSearchRetriever,
)

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context
# nltk.download('punkt')

st.title("㉄ Chat with files")

def set_reload_setting_flag():
    # st.write("New document need upload")
    st.session_state["vectorreloadflag"] = True

def set_reload_db_flag():
    st.session_state["index_db_reload_flag"] = True

def type_status_changed():
    # st.write("New document need upload")
    st.session_state["type_status_changed"] = True

def main():
    # initial parameter & LLM
    if "vectorreloadflag" not in st.session_state:
        st.session_state["vectorreloadflag"] = True
    if "index_db_reload_flag" not in st.session_state:
        st.session_state["index_db_reload_flag"] = False
    work_path = os.path.abspath('.')
    if "FileChat" not in st.session_state:
        chat = ChatBot(work_path + "/tempDir/output",
                       work_path + "/index",
                       work_path)
        st.session_state["FileChat"] = chat
    # Layout of output/setup containers
    setup_container = st.container()
    instruction_container = st.container()
    response_container = st.container()
    tab1, tab2 = st.sidebar.tabs(["Document", "Setting"])

    with st.sidebar:
        # st.sidebar.expander("Settings")
        with tab2:
            st.subheader("Parameter for document chains")
            aa_llm_model = st.selectbox(label="1.LLM Model",
                                                options=["gpt-4o-mini", "gpt-4o", "ollama"],
                                                index=0,
                                                on_change=set_reload_setting_flag)
            aa_embed_model = st.selectbox(label="2.Embedding Model",
                                        options=["text-embedding-ada-002", "ollama"],
                                        index=0,
                                        on_change=set_reload_setting_flag)
            aa_combine_type = st.radio(label="3.Types of combine document chains",
                                               options=["stuff", "map_reduce", "refine", "map_rerank"],
                                               on_change=type_status_changed)
            aa_temperature = st.selectbox(label="4.Temperature (0~1)",
                                                  options=["0", "0.2", "0.4", "0.6","0.8", "1.0"],
                                                  index=1,
                                                  on_change=set_reload_setting_flag)
            if "16k" in aa_llm_model:
                aa_max_resp_max_val = 16 * 1024
            else:
                aa_max_resp_max_val = 4096
            aa_max_resp = st.slider(label="5. Max response",
                                            min_value=256,
                                            max_value=aa_max_resp_max_val,
                                            value=2048,
                                            on_change=set_reload_setting_flag)
            if st.session_state["vectorreloadflag"] == True:
                st.session_state["FileChat"].initial_llm(aa_llm_model, aa_embed_model, aa_max_resp, float(aa_temperature))
                st.session_state["vectorreloadflag"] = False

    # main page
        with tab1:
            # with setup_container:
            aa_retriver = st.radio(label="`Choose retriever`",
                                   options=["Local Vector DB",
                                            "Azure Cognitive Search"],
                                   index=0,
                                   on_change=set_reload_setting_flag)
            # upload file & create index base
            st.subheader("Please upload your file below.")
            if "vectordb" not in st.session_state:
                st.session_state["vectordb"] = None
            file_paths = st.file_uploader("1.Upload a document file",
                                          type=["pdf", "txt", "pptx", "docx", "html"],
                                          accept_multiple_files=True)#, on_change=is_upload_status_changed)

            # if select Azure Cognitive Search or local DB
            if aa_retriver == "Azure Cognitive Search":
                st.session_state["vectordb"] = AzureCognitiveSearchRetriever(content_key="content")
            else:

                if st.button("Upload"):
                    if file_paths is not None or len(file_paths) > 0:
                        # save file
                        with st.spinner('Reading file'):
                            uploaded_paths = []
                            for file_path in file_paths:
                                uploaded_paths.append(os.path.join(work_path + "/tempDir/output", file_path.name))
                                uploaded_path = uploaded_paths[-1]
                                with open(uploaded_path, mode="wb") as f:
                                    f.write(file_path.getbuffer())
                        with st.spinner('Create vector DB'):
                            for uploaded_path in uploaded_paths:
                                tmp_vecter_db_index = st.session_state["FileChat"].setup_vectordb(uploaded_path)
                                st.session_state["vectordb"] = tmp_vecter_db_index #.as_retriever()
                                if os.path.exists(uploaded_path) == True:
                                    st.write(f"✅ {Path(uploaded_path).name} uploaed")

                # select the specified index base(s)
                index_file_list = st.session_state["FileChat"].get_all_files_list("./index", "faiss")
                options = st.multiselect('2.What file do you want to exam?',
                                         index_file_list,
                                         # [Path(file_path.name).stem],
                                         on_change=set_reload_db_flag)
                if len(options) > 0:
                    if st.session_state["index_db_reload_flag"] == True:
                        with st.spinner('Load Index DB'):
                            st.session_state["vectordb"] = st.session_state["FileChat"].load_vectordb(options)
                        # st.session_state["index_db_reload_flag"] = False
                    if (st.session_state["vectordb"] is not None):
                        st.write("✅ " + ", ".join(options) + " Index DB Loaded")
                else:
                    st.session_state["vectordb"] = None

    # Initialize chat history
    if "F_messages" not in st.session_state:
        st.session_state['F_messages'] = [{"role": "assistant", "content": {"answers": "I'm FileChat, How may I help you?", "reference": []}}]

    # Display chat messages from history on app rerun
    for message in st.session_state["F_messages"]:
        if message["role"] == "assistant":
            with st.chat_message(message["role"]):
                st.markdown(message["content"]["answers"])
                ref = message["content"]["reference"]
                src_cnt = len(ref)
                if src_cnt > 0:
                    st.markdown('Reference:')
                    for j in range(src_cnt):
                        src_file = ref[j]['source']
                        page = ref[j]['page']
                        content = ref[j]['content']
                        st.markdown(str(j) + '.' + src_file + " - page " + str(page))
                        with st.expander("See details"):
                            st.markdown(content)
        else:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Type you input here"):
        # Add user message to chat history
        st.session_state["F_messages"].append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            # prepare the setup
            if len(file_paths) > 0 or st.session_state["vectordb"] is not None:
                ## generated stores langchain chain, to enable memory function of langchain in streamlit
                if ("QA_chain" not in st.session_state) or \
                        (st.session_state["type_status_changed"] == True) or \
                        (st.session_state["index_db_reload_flag"] == True):
                    qa_chain = st.session_state["FileChat"].chat_QA_with_type_select(st.session_state["vectordb"],
                                                                                     aa_combine_type)
                    # if aa_combine_type == "stuff":
                    #     qa_chain = st.session_state["FileChat"].chat_QA(st.session_state["vectordb"])
                    # else:
                    #     qa_chain = st.session_state["FileChat"].chat_QA_map_reduce(st.session_state["vectordb"])
                    st.session_state["QA_chain"] = qa_chain
                    st.session_state["type_status_changed"] = False
                    st.session_state["index_db_reload_flag"] = False

            message_placeholder = st.empty()
            full_response = ""
            with st.spinner('preparing answer'):
                response = st.session_state["QA_chain"]({"question": prompt})
                # for response in st.session_state["QA_chain"]({"question": prompt}):
                #     full_response += response #["answer"]  # .choices[0].delta.get("content", "")
                #     time.sleep(0.001)
                #     message_placeholder.markdown(full_response + "▌")
            full_response = response["answer"]
            message_placeholder.markdown(full_response)

            # store reference
            src_cnt = len(response['source_documents'])
            ref = []
            for i in range(src_cnt):
                ref.append({})
                ref[i]['source'] = Path(response['source_documents'][i].metadata['source']).stem
                if 'page' in response['source_documents'][i].metadata.keys():
                    ref[i]['page'] = response['source_documents'][i].metadata['page']
                else:
                    ref[i]['page'] = ""
                ref[i]['content'] = response['source_documents'][i].page_content

            # print reference
            src_cnt = len(ref)
            st.markdown('Reference:')
            for j in range(src_cnt):
                src_file = ref[j]['source']
                page = ref[j]['page']
                content = ref[j]['content']
                st.markdown(str(j) + '.' + src_file + " - page " + str(page))
                with st.expander("See details"):
                    st.markdown(content)
            full_response_n_src = {"answers": full_response, "reference": ref}
        st.session_state['F_messages'].append({"role": "assistant", "content": full_response_n_src})


if __name__ == "__main__":
    main()