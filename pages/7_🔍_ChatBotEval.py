import streamlit as st
import os, time
from src.chat import ChatBot


def set_reload_setting_flag():
    # st.write("New document need upload")
    st.session_state["evalreloadflag"] = True


def main():
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
                               options=["Azure Cognitive Search", "OpenAI"],
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

if __name__ == "__main__":
    main()
