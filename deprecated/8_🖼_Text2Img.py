import streamlit as st
import os, json, requests, datetime, shutil
import openai
from openai import AzureOpenAI
from dotenv import load_dotenv

env_path = os.path.abspath('.')
st.set_page_config(page_title="Text2Img - Draw what you say")

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
        openai.api_key = os.getenv("OPENAI_API_KEY")

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
def set_reload_flag():
    # st.write("New document need upload")
    st.session_state["text2picreloadflag"] = True

def main():
    setup_env()
    st.title('🖼 Text2Img - Draw what you say')
    # Sidebar contents
    if "text2picreloadflag" not in st.session_state:
        st.session_state["text2picreloadflag"] = None
    with st.sidebar:
        st.sidebar.expander("Settings")
        st.sidebar.subheader("Parameter for Dall-E-3")
        aa_size = st.sidebar.selectbox(label="`1. Size`",
                                            options=["1024x1024", "1792x1024", "1024x1792"],
                                            index=0,
                                            on_change=set_reload_flag)
        aa_style = st.sidebar.selectbox(label="`2. Style`",
                                        options=["vivid", "natural"],
                                        index=0,
                                        on_change=set_reload_flag)
        aa_quality = st.sidebar.selectbox(label="`3. Quality`",
                                           options=["standard", "hd"],
                                           index=0,
                                           on_change=set_reload_flag)
        if "text2pic_client" not in st.session_state or st.session_state["text2picreloadflag"] == True:
            chain = initial_llm()
            st.session_state["text2pic_client"] = chain
            st.session_state["text2picreloadflag"] = False

    # Initialize chat history
    if "text2pic_messages" not in st.session_state:
        st.session_state['text2pic_messages'] = [{"role": "assistant", "content": ["I'm Text2Pic, what would you like to draw?"]}]

    # Display chat messages from history on app rerun
    for message in st.session_state["text2pic_messages"]:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                st.markdown(message["content"][0])
                if len(message["content"]) > 1:
                    st.image(message["content"][1], width=256)

    # Accept user input
    if prompt := st.chat_input("Type you input here"):
        # Add user message to chat history
        st.session_state["text2pic_messages"].append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant", avatar="🎨"):
            message_placeholder = st.empty()
            full_response = ""
            # Set the directory for the stored image
            image_dir = os.path.join(os.curdir, 'images')
            # If the directory doesn't exist, create it
            if not os.path.isdir(image_dir):
                os.mkdir(image_dir)
            nowTime = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            image_path = os.path.join(image_dir, 'generated_image' + nowTime + '.png')
            with st.spinner('drawing the picture'):
                full_response = st.session_state["text2pic_client"].images.generate(
                                                                                    model="Dalle3", # the name of your DALL-E 3 deployment
                                                                                    prompt=prompt, #"落霞与孤鹜齐飞，秋水共长天一色",#"a close-up of a bear walking through the forest",
                                                                                    size=aa_size,
                                                                                    style=aa_style, #"vivid", "natural"
                                                                                    quality=aa_quality, #"standard" "hd"
                                                                                    n=1
                                                                                )
                json_response = json.loads(full_response.model_dump_json())
                revised_prompt = json_response["data"][0]["revised_prompt"]
                # Retrieve the generated image
                image_url = json_response["data"][0]["url"]  # extract image URL from response
            st.markdown(revised_prompt)
            message_placeholder.image(image_url)
            with st.spinner('Prepare the download button'):
                generated_image = requests.get(image_url).content  # download the image
                with open(image_path, "wb") as image_file:
                    image_file.write(generated_image)
                with open(image_path, "rb") as file:
                    btn = st.download_button(
                        label="Download image",
                        data=file,
                        # file_name="flower.png",
                        mime="image/png"
                    )

        st.session_state['text2pic_messages'].append({"role": "assistant", "content": [revised_prompt, image_path]})


if __name__ == "__main__":
    main()