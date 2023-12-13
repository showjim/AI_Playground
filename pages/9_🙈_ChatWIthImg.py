import time

import streamlit as st
import os, base64, glob, shutil, json
from pathlib import Path
import openai
from openai import AzureOpenAI
from dotenv import load_dotenv

st.title("🙈 Chat with IMG")

def setup_env():
    # Load OpenAI key
    if os.path.exists("key.txt"):
        shutil.copyfile("key.txt", ".env")
        load_dotenv()
    else:
        print("key.txt with OpenAI API is required")

    # Load config values
    if os.path.exists(os.path.join(r'config.json')):
        with open(r'config.json') as config_file:
            config_details = json.load(config_file)

        # Setting up the embedding model
        embedding_model_name = config_details['EMBEDDING_MODEL']
        openai.api_type = "azure"
        openai.api_base = config_details['OPENAI_API_BASE']
        openai.api_version = config_details['OPENAI_API_VERSION']
        openai.azure_endpoint = os.getenv("OPENAI_API_KEY")

        # Dalle-E-3
        os.environ["AZURE_OPENAI_API_KEY_SWC"] = os.getenv("AZURE_OPENAI_API_KEY_SWC")
        os.environ["AZURE_OPENAI_ENDPOINT_SWC"] = config_details['AZURE_OPENAI_ENDPOINT_SWC']
    else:
        print("config.json with Azure OpenAI config is required")

def initial_llm():
    client = AzureOpenAI(
        api_version="2023-12-01-preview",
        api_key=os.environ["AZURE_OPENAI_API_KEY_SWC"],
        azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT_SWC']
    )
    return client
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
        file_name = Path(filepath).name
        result.append(file_name)
    return result

def main():
    setup_env()
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
    if "IMGChatChain" not in st.session_state:
        chain = initial_llm()
        st.session_state["IMGChatChain"] = chain
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
                                max_value=2048,
                                value=300,
                                on_change=set_reload_setting_flag)
        aa_detail = st.selectbox(label="3. Detail",
                                 options=["low", "high"],
                                 index=0,
                                 on_change=set_reload_setting_flag)
        if st.session_state["SettingReloadFlag"] == True:
            if "IMGChatSetting" not in st.session_state:
                st.session_state["IMGChatSetting"] = {}
            st.session_state["IMGChatSetting"] = {"model":aa_llm_model, "max_tokens": aa_max_resp,
                                                  "temperature": aa_temperature, "detail": aa_detail}
        # with setup_container:
        # upload file & create index base
        st.subheader("Please upload your file below.")
        if "IMGDB" not in st.session_state:
            st.session_state["IMGDB"] = None
        file_path = st.file_uploader("1.Upload a document file",
                                      type=["jpg", "png", "gif", "bmp"],
                                      accept_multiple_files=False)  # , on_change=is_upload_status_changed)

        if st.button("Upload"):
            if file_path is not None:
                # save file
                with st.spinner('Reading file'):
                    uploaded_path = os.path.join(work_path + "/img", file_path.name)
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
                with st.spinner('Load Image File'):
                    IMAGE_PATH = os.path.join("./img", options[0])
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
                                    "detail": st.session_state["IMGChatSetting"]["detail"]
                                }
                            }
                        ]
                    }
                    st.session_state["ImgChatMessages"].append(tmp_usr_img_msg)
                st.write("✅ " + ", ".join(options) + " IMG Loaded")
            if st.session_state["IMGDB"] is None:
                print("can you reach here?")
        # else:
        #     st.session_state["IMGDB"] = {}

    # Display chat messages from history on app rerun
    for message in st.session_state["ImgChatMessages"]:
        if message["role"] == "assistant":
            with st.chat_message(message["role"]):
                st.markdown(message["content"][0]["text"])
        else:
            with st.chat_message(message["role"]):
                for content in message["content"]:
                    if content["type"] == "image_url":
                        image_url = content["image_url"]["url"].replace("data:image/jpeg;base64,", "")
                        img_paths = get_keys(st.session_state["IMGDB"], image_url)
                        st.image(img_paths[0])
                    elif content["type"] == "text":
                        st.markdown(content["text"])
                    else:
                        print("Error: Unexcept TYPE Found: " + content["type"])

    # Accept user input
    if prompt := st.chat_input("Type you input here"):
        # Add user message to chat history
        st.session_state["ImgChatMessages"].append({"role": "user", "content": [{"type": "text", "text": prompt}]})
        print("USER: " + prompt)
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            # # prepare the setup
            # if len(file_paths) > 0 or st.session_state["IMGDB"] is not None:
            message_placeholder = st.empty()
            full_response = ""
            with st.spinner('preparing answer'):
                response = st.session_state["IMGChatChain"].chat.completions.create(
                    model=st.session_state["IMGChatSetting"]["model"],
                    messages=st.session_state['ImgChatMessages'],
                    max_tokens=st.session_state["IMGChatSetting"]["max_tokens"],
                    # default max tokens is low so set higher
                    temperature=st.session_state["IMGChatSetting"]["temperature"],
                    stream=True
                )
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content #["answer"]  # .choices[0].delta.get("content", "")
                    time.sleep(0.001)
                    message_placeholder.markdown(full_response + "▌")
            # full_response = response.choices[0].message.content
            print("AI: " + full_response)
            message_placeholder.markdown(full_response)
            full_response_list = [{"type": "text", "text": full_response}]
        st.session_state['ImgChatMessages'].append({"role": "assistant", "content": full_response_list})


if __name__ == "__main__":
    main()
