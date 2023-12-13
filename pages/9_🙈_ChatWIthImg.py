import time

import streamlit as st
import os, base64, glob
from pathlib import Path
from src.chat import ChatBot, StreamHandler
from langchain.retrievers import (
    AzureCognitiveSearchRetriever,
)

st.title("🙈 Chat with IMG")
def set_reload_setting_flag():
    # st.write("New document need upload")
    st.session_state["SettingReloadFlag"] = True

def set_reload_img_flag():
    # st.write("New document need upload")
    st.session_state["ImgReloadFlag"] = True

def get_keys(d, value):
    return [k for k,v in d.items() if v == value]

def get_all_files_list(source_dir, exts):
    all_files = []
    result = []
    for ext in exts:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"*.{ext}"), recursive=False)
        )
    for filepath in all_files:
        file_basename = Path(filepath).stem
        result.append(file_basename)
    return result

def main():
    # initial parameter & LLM
    if "ImgReloadFlag" not in st.session_state:
        st.session_state["ImgReloadFlag"] = True
    if "SettingReloadFlag" not in st.session_state:
        st.session_state["SettingReloadFlag"] = True
    if "IMGDB" not in st.session_state:
        st.session_state["IMGDB"] = {}
    # Initialize chat history
    if "ImgChatMessages" not in st.session_state:
        st.session_state['ImgChatMessages'] = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Hi, I'm ImgChatBot, please load the IMG you want to chat.",
                    }
                ]
            }
        ]
    work_path = os.path.abspath('.')

    with st.sidebar:
        st.sidebar.expander("Settings")

        st.subheader("Parameter for document chains")
        aa_llm_model = st.selectbox(label="0.LLM Model",
                                    options=["gpt-4-vision-preview"],
                                    index=0,
                                    on_change=set_reload_setting_flag)
        aa_temperature = st.selectbox(label="1.Temperature (0~1)",
                                      options=["0", "0.2", "0.4", "0.6", "0.8", "1.0"],
                                      index=1,
                                      on_change=set_reload_setting_flag)
        aa_max_resp = st.slider(label="2. Max response",
                                min_value=256,
                                max_value=300,
                                value=2048,
                                on_change=set_reload_setting_flag)

        # with setup_container:
        # upload file & create index base
        st.subheader("Please upload your file below.")
        if "IMGDB" not in st.session_state:
            st.session_state["IMGDB"] = None
        file_paths = st.file_uploader("1.Upload a document file",
                                      type=["jpg", "png", "gif", "bmp"],
                                      accept_multiple_files=False)  # , on_change=is_upload_status_changed)

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
                        if os.path.exists(uploaded_path) == True:
                            st.write(f"✅ {Path(uploaded_path).name} uploaed")

        # select the specified index base(s)
        index_file_list = get_all_files_list("./img", ["jpg", "png", "gif", "bmp"])
        options = st.multiselect('2.What img do you want to exam?',
                                 index_file_list,
                                 max_selections=1,
                                 on_change=set_reload_img_flag)
        if len(options) > 0:
            if st.session_state["ImgReloadFlag"] == True:
                st.session_state["ImgReloadFlag"] = False
                with st.spinner('Load IMG'):
                    IMAGE_PATH = options[0]
                    encoded_image = base64.b64encode(open(IMAGE_PATH, 'rb').read()).decode('utf-8')

                    # store in dict for history
                    st.session_state["IMGDB"][IMAGE_PATH] = encoded_image

                    # add in chat msg
                    tmp_usr_img_msg = {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encoded_image}",
                                    "detail": "low"
                                }
                            }
                        ]
                    }
                    st.session_state["ImgChatMessages"].append(tmp_usr_img_msg)
                st.write("✅ " + ", ".join(options) + " IMG Loaded")
            if st.session_state["IMGDB"] is not None:
                print("can you reach here?")
        # else:
        #     st.session_state["IMGDB"] = {}

    # Display chat messages from history on app rerun
    for message in st.session_state["ImgChatMessages"]:
        if message["role"] == "assistant":
            with st.chat_message(message["role"]):
                st.markdown(message["content"]["text"])
        else:
            with st.chat_message(message["role"]):
                if len(message["content"]) == 1:
                    st.markdown(message["content"][0]["text"])
                else:
                    image_url = message["content"][0]["image_url"].replace("data:image/jpeg;base64,", "")
                    img_paths = get_keys(st.session_state["IMGDB"], image_url)
                    st.image(img_paths[0])
                    st.markdown(message["content"][1]["text"])


    # Accept user input
    if prompt := st.chat_input("Type you input here"):
        # Add user message to chat history
        st.session_state["ImgChatMessages"].append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            # prepare the setup
            if len(file_paths) > 0 or st.session_state["IMGDB"] is not None:
                ## generated stores langchain chain, to enable memory function of langchain in streamlit


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
            full_response_n_src = {"answers": full_response, "ImgForChat": ref}
        st.session_state['ImgChatMessages'].append({"role": "assistant", "content": full_response_n_src})


if __name__ == "__main__":
    main()
