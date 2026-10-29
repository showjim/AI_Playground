import streamlit as st
import os, time, json, io, base64, requests, sqlite3
from typing import List
import azure.cognitiveservices.speech as speechsdk
from src.ClsChatBot import ChatRobotAzure, ChatRobotOpenRouter, ChatRobotSiliconFlow
from src import memory
from pathlib import Path

# __version__ = "Beta V0.0.2"
env_path = os.path.abspath(".")
CHAT_HISTORY_PATH = "./setting/chat_history_nangua.db"
chatbot_azure = ChatRobotAzure()
chatbot_azure.setup_env()
chatbot_or = ChatRobotOpenRouter()
chatbot_or.setup_env()
client = chatbot_or.initial_llm() #chatbot.initial_llm()
chatbot_siliconflow = ChatRobotSiliconFlow()
# client_dalle3 = chatbot_azure.initial_dalle3()
# client_stt = chatbot_azure.initial_whisper()
chatbot_azure.initial_stt()
tools = chatbot_or.initial_tools()


st.set_page_config(page_title="FreeChat - Chatbot With Native APIs")

# Database connection context manager
class DatabaseConnection:
    def __init__(self, db_name):
        self.db_name = db_name

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_name, check_same_thread=False)
        self.cursor = self.conn.cursor()
        return self.cursor

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.commit()
            self.conn.close()

# Initialize database
def init_database():
    with DatabaseConnection(CHAT_HISTORY_PATH) as c:
        # Add indexes for better performance
        c.execute('''CREATE TABLE IF NOT EXISTS topics
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    name TEXT UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        c.execute('''CREATE TABLE IF NOT EXISTS chats
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    topic_id INTEGER, 
                    role TEXT, 
                    type TEXT, 
                    message TEXT,
                    img TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(topic_id) REFERENCES topics(id) ON DELETE CASCADE)''')
        # Add indexes
        c.execute('''CREATE INDEX IF NOT EXISTS idx_topic_id ON chats(topic_id)''')
        c.execute('''CREATE INDEX IF NOT EXISTS idx_created_at ON chats(created_at)''')
    memory.init_db(CHAT_HISTORY_PATH)

# Initialize session state
def init_session_state():
    if "current_topic_id" not in st.session_state:
        st.session_state.current_topic_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "new_topic" not in st.session_state:
        st.session_state.new_topic = ""
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = ""

def create_topic(system_prompt:str):
    if st.session_state.new_topic:
        try:
            with DatabaseConnection(CHAT_HISTORY_PATH) as c:
                c.execute("INSERT INTO topics (name) VALUES (?)", (st.session_state.new_topic,))
                topic_id = c.lastrowid
            memory.init_topic(CHAT_HISTORY_PATH, topic_id)
            st.session_state.current_topic_id = topic_id
            st.session_state.messages = []
            # Clear the input
            st.session_state.new_topic = ""
            st.session_state.FreeChatMessages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
            st.session_state.FreeChatMessagesDisplay = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}]

            save_chat_to_db(st.session_state.current_topic_id, "system", "text", system_prompt)
        except sqlite3.IntegrityError:
            st.sidebar.error("Topic name already exists!")
        except Exception as e:
            st.sidebar.error(f"Error creating topic: {str(e)}")

# Save message to database
def save_chat_to_db(current_topic_id:int, role:str, type:str, message:str, img:str=""):
    with DatabaseConnection(CHAT_HISTORY_PATH) as c:
        c.execute(
            "INSERT INTO chats (topic_id, role, type, message, img) VALUES (?, ?, ?, ?, ?)",
            (current_topic_id, role, type, message, img)
        )


def build_api_messages(prompt: str) -> List[dict]:
    """Copy the history and add long-term memory to the system message.

    The memory block is never written back to FreeChatMessages or the database.
    """
    messages = list(st.session_state.FreeChatMessages)
    try:
        context = memory.build_memory_context(CHAT_HISTORY_PATH, prompt)
    except Exception as e:
        print(f"Memory context failed: {e}")
        context = ""
    if context and messages and messages[0].get("role") == "system":
        system_message = dict(messages[0])
        content = system_message.get("content")
        if isinstance(content, list) and content:
            system_text = content[0].get("text", "")
        elif isinstance(content, str):
            system_text = content
        else:
            system_text = ""
        system_message["content"] = [
            {"type": "text", "text": system_text + "\n\n" + context}
        ]
        messages[0] = system_message
    return messages

def set_reload_mode():
    st.session_state.FreeChatReloadMode = True
    st.session_state.FreeChatModeChanged = True


def set_reload_flag():
    # st.write("New document need upload")
    st.session_state["FreeChatReloadFlag"] = True

def set_reload_img_flag():
    # st.write("New document need upload")
    st.session_state["FreeChatImgReloadFlag"] = True

def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": unit})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


def create_img_by_dalle3(prompt):
    """Create image by call to Dall-E3"""
    # result = client_dalle3.images.generate(
    #     model="Dalle3",  # the name of your DALL-E 3 deployment
    #     prompt=prompt,  # "a close-up of a bear walking through the forest",
    #     size="1024x1024",
    #     style="vivid",  # "vivid", "natural"
    #     # quality="auto",  # "standard" "hd"
    #     n=1
    # )
    # json_response = json.loads(result.model_dump_json())
    # # Retrieve the generated image
    # image_url = json_response["data"][0]["url"]  # extract image URL from response
    # revised_prompt = json_response["data"][0]["revised_prompt"]
    # print("Dall-E3: " + revised_prompt)
    # print("Dall-E3: " + image_url)
    # return image_url

def create_img_from_siliconflow(prompt:str):
    """Create image by call to SiliconFlow"""
    headers = {
        "Authorization": f"Bearer {os.getenv('SILICONFLOW_API_KEY')}",
        "Content-Type": "application/json"
    }

    url = "https://api.siliconflow.cn/v1/images/generations"

    payload = {
        "model": "Qwen/Qwen-Image", # "Kwai-Kolors/Kolors"
        "prompt": prompt,
        "negative_prompt": "<string>",
        "image_size": "1472x1140",
        "batch_size": 1,
        "seed": 4999999999,
        "num_inference_steps": 20,
        "guidance_scale": 7.5
    }

    response = requests.post(url, json=payload, headers=headers)

    print(response.text)

    json = response.json()
    if "images" not in json:
        return json["message"]
    url = json["images"][0]["url"]
    return url

def execute_function_call(available_functions, tool_call):
    function_name = tool_call["function"]["name"]
    function_to_call = available_functions.get(function_name, None)
    if function_to_call:
        function_args = json.loads(tool_call["function"]["arguments"])
        function_response = function_to_call(**function_args)
    else:
        function_response = f"Error: function {function_name} does not exist"
    return function_response


def whisper_STT(audio_test_file, audio_language="en", prompt="以下是普通话的句子。", translate=False):
    model_name = "whisper-1"
    result = ""
    # if translate:
    #     result = client_stt.audio.translations.create(
    #         file=audio_test_file,  # open(audio_test_file, "rb"),
    #         model=model_name,
    #         response_format="text",
    #     )
    # else:
    #     result = client_stt.audio.transcriptions.create(
    #                 file=audio_test_file, #open(audio_test_file, "rb"),
    #                 model=model_name,
    #                 language=audio_language,
    #                 response_format="text",
    #                 prompt=prompt,
    #     )
    # return result

@st.cache_resource
def create_img_store_dict(index_file_list: List[str]):
    st.session_state.FreeIMGDB = {os.path.join("./img", file):base64.b64encode(open(os.path.join("./img", file), 'rb').read()).decode('utf-8') for file in index_file_list}

def main():
    index = 0
    st.title("🎙️Free Chat Web-UI App")

    init_database()

    # Sidebar contents
    if "FreeChatReloadMode" not in st.session_state:
        st.session_state["FreeChatReloadMode"] = True
    if "FreeChatReloadFlag" not in st.session_state:
        st.session_state["FreeChatReloadFlag"] = True
    if "AvatarImg" not in st.session_state:
        st.session_state["AvatarImg"] = None
    # Initialize chat history
    if "FreeChatMessages" not in st.session_state:
        st.session_state["FreeChatMessages"] = []
    if "FreeChatMessagesDispay" not in st.session_state:
        # this is a shadow of "FreeChatMessages" to keep image URL from Dalle3
        st.session_state["FreeChatMessagesDispay"] = []
    # chain = chatbot.initial_llm()
    if "FreeChatChain" not in st.session_state:
        # client = chatbot.initial_llm()
        st.session_state.FreeChatChain = client
    if "FreeIMGDB" not in st.session_state:
        st.session_state["FreeIMGDB"] = {}
    if "FreeChatImgReloadFlag" not in st.session_state:
        st.session_state["FreeChatImgReloadFlag"] = False
    init_session_state()
    work_path = os.path.abspath('.')

    with st.sidebar:

        st.subheader("1. Settings")
        with st.expander("Parameters for Chatbot"):

            aa_chat_mode = st.selectbox(label="`0. Chat Mode`",
                                                options=["CasualChat", "Meta-prompt", "Translate", "西瓜一家-小南瓜", "西瓜一家-小东瓜",
                                                         "西瓜一家-Ana"],
                                                index=0,
                                                on_change=set_reload_mode)
            aa_llm_model = st.selectbox(label="`1. LLM Model`",
                                                options=[
                                                    "deepseek/deepseek-v4.1-flash-1017",
                                                    "deepseek/deepseek-v4-pro",
                                                    "moonshotai/kimi-k2.6",
                                                    "qwen/qwen3.6-plus",
                                                    "google/gemini-3.1-pro-preview",
                                                    "bytedance-seed/seed-1.6-flash"
                                                ],
                                                index=0,
                                                on_change=set_reload_flag)
            aa_temperature = st.selectbox(label="`2. Temperature (0~1)`",
                                                  options=["0", "0.2", "0.4", "0.6", "0.8", "1.0"],
                                                  index=1,
                                                  on_change=set_reload_flag)
            aa_max_resp = st.slider(label="`3. Max response`",
                                            min_value=1024,
                                            max_value=16 * 1024,
                                            value=4096,
                                            on_change=set_reload_flag)
            aa_context_msg = st.select_slider(label="`4. Context message`",
                                                      options=[1, 5, 10, 20, 50, 100],
                                                      value=10,
                                                      on_change=set_reload_flag
                                                      )

            if st.session_state.FreeChatReloadMode:
                st.session_state.system_prompt = chatbot_or.select_chat_mode(aa_chat_mode)
                st.session_state.FreeChatReloadMode = False
                # Apply a mode change to the current topic; loading a topic leaves its stored prompt alone
                if st.session_state.get("FreeChatModeChanged", False):
                    st.session_state.FreeChatModeChanged = False
                    system_msg = {"role": "system",
                                  "content": [{"type": "text", "text": st.session_state.system_prompt}]}
                    for msg_list in (st.session_state.FreeChatMessages,
                                     st.session_state.get("FreeChatMessagesDisplay", [])):
                        if msg_list and msg_list[0].get("role") == "system":
                            msg_list[0] = system_msg
                    if st.session_state.current_topic_id:
                        with DatabaseConnection(CHAT_HISTORY_PATH) as c:
                            c.execute("UPDATE chats SET message = ? WHERE topic_id = ? AND role = 'system'",
                                      (st.session_state.system_prompt, st.session_state.current_topic_id))
                # set the tool choice in fuction call
                if aa_chat_mode == "Translate":
                    st.session_state["tool_choice"] = "none"
                else:
                    st.session_state["tool_choice"] = "auto"
                # initial the avatar and greeting
                if aa_chat_mode == "西瓜一家-小南瓜":
                    st.session_state["AvatarImg"] = "./img/Sunny.png"
                else:
                    st.session_state["AvatarImg"] = "assistant"
                # st.session_state.FreeChatMessages = [
                #     {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                #     {"role": "assistant", "content": [{"type": "text", "text": initial_msg}]}
                # ]
                # st.session_state.FreeChatMessagesDisplay = [
                #     {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                #     {"role": "assistant", "content": [{"type": "text", "text": initial_msg}]}
                # ]
                # Save user message to database
                # save_chat_to_db(st.session_state.current_topic_id, "system", "text", system_prompt)
            if st.session_state["FreeChatReloadFlag"] == True:
                if "FreeChatSetting" not in st.session_state:
                    st.session_state["FreeChatSetting"] = {}
                st.session_state["FreeChatSetting"] = {"model": aa_llm_model, "max_tokens": aa_max_resp,
                                                       "temperature": float(aa_temperature), "context_msg": aa_context_msg}
                st.session_state["FreeChatReloadFlag"] = False

            # Text2Speech
            aa_voice_name = st.selectbox(label="`5. Voice Name`",
                                                 options=["None", "小南瓜", "小东瓜", "Ana"],
                                                 index=0)
            chatbot_azure.speech_config.speech_recognition_language = "zh-CN"  # "zh-CN" #"en-US"
            if aa_voice_name == "小南瓜":
                aa_voice_name = "zh-CN-XiaoyiNeural"
            elif aa_voice_name == "小东瓜":
                aa_voice_name = "zh-CN-YunxiaNeural"
            elif aa_voice_name == "Ana":
                aa_voice_name = "en-US-AnaNeural"
                chatbot_azure.speech_config.speech_recognition_language = "en-US"  # "zh-CN" #"en-US"

        with st.expander("Topics"):
            # New topic input with callback
            st.text_input(
                "New Topic",
                placeholder="Enter topic name",
                key="new_topic",
                on_change=create_topic(st.session_state.system_prompt)
            )

            # Display topics
            with DatabaseConnection(CHAT_HISTORY_PATH) as c:
                c.execute("SELECT id, name FROM topics ORDER BY created_at DESC")
                topics = c.fetchall()

            if topics:
                st.divider()
                for topic_id, topic_name in topics:
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        if st.button(f"📝 {topic_name}", key=f"topic_{topic_id}"):
                            previous_topic_id = st.session_state.current_topic_id
                            if (
                                previous_topic_id
                                and previous_topic_id != topic_id
                                and memory.has_new_messages(
                                    CHAT_HISTORY_PATH, previous_topic_id
                                )
                            ):
                                with st.spinner("小南瓜在回想刚才的聊天..."):
                                    try:
                                        memory.maybe_extract_memories(
                                            CHAT_HISTORY_PATH,
                                            previous_topic_id,
                                            st.session_state.FreeChatChain,
                                            st.session_state.FreeChatSetting["model"],
                                            force=True,
                                        )
                                    except Exception as e:
                                        print(f"Memory extraction failed: {e}")
                            st.session_state.FreeChatReloadMode = True
                            st.session_state.current_topic_id = topic_id
                            # Load messages for this topic
                            with DatabaseConnection(CHAT_HISTORY_PATH) as c:
                                c.execute("SELECT role,type,message,img FROM chats WHERE topic_id = ? ORDER BY created_at",
                                          (topic_id,))
                                db_content = c.fetchall()

                                st.session_state.FreeChatMessages = []
                                st.session_state.FreeChatMessagesDisplay = []
                                for m in db_content:
                                    if m[1] != 'image_url':
                                        if m[3].startswith("https://"):
                                            st.session_state.FreeChatMessages.append(
                                                {"role": m[0], "content": [{"type": m[1], "text":m[2]}]})
                                            st.session_state.FreeChatMessagesDisplay.append(
                                                {"role": m[0], "content": [{"type": m[1], "text":m[2]}], "image": m[3]})
                                        else:
                                            st.session_state.FreeChatMessages.append(
                                                {"role": m[0], "content": [{"type": m[1], "text": m[2]}]})
                                            st.session_state.FreeChatMessagesDisplay.append(
                                                {"role": m[0], "content": [{"type": m[1], "text": m[2]}]})
                                    else:
                                        if aa_llm_model not in [
                                            "deepseek/deepseek-v3.1-terminus",
                                            "deepseek/deepseek-chat-v3-0324",
                                            "moonshotai/kimi-k2"
                                        ]:
                                            st.session_state.FreeChatMessages.append(
                                                {"role": m[0], "content": [{"type": m[1], "image_url": {"url":m[2]}}]})
                                            st.session_state.FreeChatMessagesDisplay.append(
                                                {"role": m[0], "content": [{"type": m[1], "image_url": {"url":m[2]}}]})


                    with col2:
                        if st.button("🗑️", key=f"delete_{topic_id}"):
                            with DatabaseConnection(CHAT_HISTORY_PATH) as c:
                                c.execute("DELETE FROM topics WHERE id = ?", (topic_id,))
                            if st.session_state.current_topic_id == topic_id:
                                st.session_state.current_topic_id = None
                                st.session_state.FreeChatMessages = []
                                st.session_state.FreeChatMessagesDisplay = []
                            st.rerun()

        with st.expander("📖 小南瓜记得的事"):
            memories = memory.list_memories(CHAT_HISTORY_PATH)
            counts = memory.memory_counts(CHAT_HISTORY_PATH)
            if not memories:
                st.caption("还没有记忆。聊满 10 轮，或点“让小南瓜记住这段”。")
            else:
                st.caption(
                    f"共 {counts['total']} 条 · 本周学习 {counts['learning_week']} 条 · "
                    f"需要关注 {counts['attention']} 条 · 心情 30 天后淡出"
                )
                for kind in ("story", "learning", "life", "mood"):
                    kind_items = [m for m in memories if m["kind"] == kind]
                    if not kind_items:
                        continue
                    st.markdown(f"**{memory.KIND_LABELS[kind]}**")
                    for item in kind_items:
                        flag = f" ⚠️ {item['flag']}" if item.get("flag") else ""
                        st.markdown(f"**{item['key']}**：{item['content']}{flag}")
                        with st.popover(
                            "编辑 / 忘掉", key=f"mem_popover_{item['id']}"
                        ):
                            new_content = st.text_area(
                                "内容",
                                item["content"],
                                key=f"mem_text_{item['id']}",
                            )
                            if st.button("保存", key=f"mem_save_{item['id']}"):
                                memory.update_memory_content(
                                    CHAT_HISTORY_PATH, item["id"], new_content
                                )
                                st.rerun()
                            if st.button("忘掉", key=f"mem_forget_{item['id']}"):
                                memory.delete_memory(CHAT_HISTORY_PATH, item["id"])
                                st.rerun()
                with st.form("clear_memory_form"):
                    confirmed = st.checkbox("我确认要清空全部记忆")
                    if st.form_submit_button("清空全部记忆") and confirmed:
                        memory.clear_memories(CHAT_HISTORY_PATH)
                        st.rerun()

        st.subheader("2. STT")
        speech_txt = ""
        tab1, tab2, tab3, tab4 = st.tabs([ "SiliconFlow", "Azure STT File", "Azure STT", "Whisper"])
        with tab1:
            audio_siliconflow = st.audio_input("Record a voice message", key ="SiliconFlow", label_visibility="hidden")
            if audio_siliconflow and st.session_state.get("FreeChatLastAudioSiliconFlow") != audio_siliconflow.file_id:
                # Since Azure Whisper cannot be used any more so...
                # I have to switch to SiliconFlow STT
                speech_txt = chatbot_siliconflow.stt(audio_siliconflow)
                st.session_state["FreeChatLastAudioSiliconFlow"] = audio_siliconflow.file_id
        with tab2:
            audio_azure = st.audio_input("Record a voice message", key ="Azure STT File", label_visibility="hidden")
            if audio_azure and st.session_state.get("FreeChatLastAudioAzure") != audio_azure.file_id:
                # Since Azure Whisper cannot be used any more so...
                # I have to switch to Azure STT
                filename = "./tmp.wav"
                with open(filename, "wb") as f:
                    f.write(audio_azure.getvalue())
                speech_txt = chatbot_azure.speech_2_text_continuous_file_based(filename) # speech_2_text_file_based(filename)
                st.session_state["FreeChatLastAudioAzure"] = audio_azure.file_id
        with tab3:
            # Speech2Text
            aa_audio_mode = st.selectbox(label="`Audio Input Mode`",
                                                 options=["Single", "Continuous"],
                                                 index=0)
            if st.button("`Speak`"):
                if aa_audio_mode == "Single":
                    speech_txt = chatbot_azure.speech_2_text()
                elif aa_audio_mode == "Continuous":
                    speech_txt = chatbot_azure.speech_2_text_continous() #speech_2_text() #speech_2_text_continous() #speech_2_text()
                else:
                    print("")
                    # speech_txt = chatbot.speech_2_text_file_based()
        # another STT: Whisper option
        with tab4:
            aa_whisper_mode = st.selectbox(label="`Whipser/Azure STT Mode`",
                                         options=["Transcribe", "Translate"],
                                         index=0)
            audio = st.audio_input("Record a voice message", key ="Whisper", label_visibility="hidden")
            if audio:
                if aa_whisper_mode == "Transcribe":
                    is_translate = False
                else:
                    is_translate = True
                speech_txt = whisper_STT(audio, "zh",translate=is_translate)

        # upload image file & create index base
        st.subheader("3. Vision")
        with st.expander("Upload your image"):
            file_path = st.file_uploader("1.Upload a image file",
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
            index_file_list = chatbot_or.get_all_files_list("./img", ["jpg", "png", "gif", "bmp"])
            create_img_store_dict(index_file_list)
            options = st.multiselect('2.What img do you want to exam?',
                                     index_file_list,
                                     max_selections=1,
                                     on_change=set_reload_img_flag)
            if len(options) > 0:
                if st.session_state.FreeChatImgReloadFlag == True:
                    st.session_state.FreeChatImgReloadFlag = False
                    with st.spinner('Load Image File'):
                        IMAGE_PATH = os.path.join("./img", options[0])
                        encoded_image = base64.b64encode(open(IMAGE_PATH, 'rb').read()).decode('utf-8')

                        # store in dict for history
                        #st.session_state.FreeIMGDB[IMAGE_PATH] = encoded_image

                        # add in chat msg
                        tmp_usr_img_msg = {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{encoded_image}",
                                        # "detail": st.session_state["IMGChatSetting"]["detail"]
                                    }
                                }
                            ]
                        }
                        st.session_state.FreeChatMessages.append(tmp_usr_img_msg)
                        st.session_state.FreeChatMessagesDisplay.append(tmp_usr_img_msg)
                        # Save user message to database
                        save_chat_to_db(st.session_state.current_topic_id, "user", "image_url", f"data:image/jpeg;base64,{encoded_image}")
                    st.write("✅ " + ", ".join(options) + " IMG Loaded")
                if st.session_state.FreeIMGDB is None:
                    print("can you reach here?")

    # Main chat area
    if st.session_state.current_topic_id:
        # Get topic name
        with DatabaseConnection(CHAT_HISTORY_PATH) as c:
            c.execute("SELECT name FROM topics WHERE id = ?", (st.session_state.current_topic_id,))
            topic_name = c.fetchone()[0]

        st.header(f"Topic: {topic_name}")

        # Display chat messages from history on app rerun
        for message in st.session_state.FreeChatMessagesDisplay:
            if message["role"] == "user":
                with st.chat_message(message["role"]):
                    for content in message["content"]:
                        if content["type"] == "image_url":
                            image_url = content["image_url"]["url"].replace("data:image/jpeg;base64,", "")
                            img_paths = chatbot_or.get_keys(st.session_state.FreeIMGDB, image_url)
                            st.image(img_paths[0])
                        elif content["type"] == "text":
                            st.markdown(content["text"])
                        else:
                            print("Error: Unexcept TYPE Found: " + content["type"])
            elif message["role"] == "assistant":
                if message["content"] is not None:
                    with st.chat_message(name=message["role"], avatar=st.session_state["AvatarImg"]):
                        for content in message["content"]:
                            st.markdown(content["text"])
                            st.button(label="Play", key="history" + str(index), on_click=chatbot_azure.text_2_speech,
                                      args=(str(content["text"]).replace("*","").replace("#", ""), aa_voice_name,))
                            index += 1
                        if "image" in message.keys():
                            st.image(message["image"], width=256)

        if st.button(
            "📝 让小南瓜记住这段",
            help="把小南瓜刚才和你说的话记进故事书、生活、心情或学习里。",
        ):
            try:
                with st.spinner("小南瓜在记..."):
                    result = memory.maybe_extract_memories(
                        CHAT_HISTORY_PATH,
                        st.session_state.current_topic_id,
                        st.session_state.FreeChatChain,
                        st.session_state.FreeChatSetting["model"],
                        force=True,
                        window=20,
                    )
            except Exception as e:
                print(f"Memory extraction failed: {e}")
                result = {"error": str(e)}
            if result.get("extracted"):
                st.toast(f"记住啦，{result['extracted']} 条。")
            elif result.get("error"):
                st.toast("这次没记住，等一下再试。")
            else:
                st.toast("这段已经记过了。")

        # Accept user input
        if (prompt := st.chat_input("Type you input here")) or (prompt := speech_txt):
            if st.session_state.current_topic_id and memory.has_due_memories(
                CHAT_HISTORY_PATH, st.session_state.current_topic_id
            ):
                with st.spinner("小南瓜在回想之前聊过的事..."):
                    try:
                        memory.maybe_extract_memories(
                            CHAT_HISTORY_PATH,
                            st.session_state.current_topic_id,
                            st.session_state.FreeChatChain,
                            st.session_state.FreeChatSetting["model"],
                        )
                    except Exception as e:
                        print(f"Memory extraction failed: {e}")
            # Add user message to chat history
            max_cnt = st.session_state["FreeChatSetting"]["context_msg"]
            st.session_state.FreeChatMessages = chatbot_or.control_msg_history_szie(st.session_state.FreeChatMessages, max_cnt)
            model_len_before_input = len(st.session_state.FreeChatMessages)
            st.session_state.FreeChatMessages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
            st.session_state.FreeChatMessagesDisplay.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
            # Save user message to database
            save_chat_to_db(st.session_state.current_topic_id, "user", "text", prompt)
            print("HUMAN: " + prompt)
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
            # Display assistant response in chat message container
            with st.chat_message("assistant", avatar=st.session_state["AvatarImg"]):
                message_placeholder = st.empty()
                btn_placeholder = st.empty()
                full_response = ""
                function_response = ""
                tool_calls = []
                cur_func_call = {"id": None, "type": "function", "function": {"arguments": "", "name": None}}
                with st.spinner("preparing answer"):  # st.session_state["FreeChatChain"]
                    try:
                        response = st.session_state.FreeChatChain.chat.completions.create(
                            model=st.session_state.FreeChatSetting["model"],
                            messages=build_api_messages(prompt),
                            max_tokens=st.session_state.FreeChatSetting["max_tokens"],
                            # default max tokens is low so set higher
                            temperature=st.session_state.FreeChatSetting["temperature"],
                            stream=True,
                            tools=tools,
                            tool_choice=st.session_state.tool_choice,  # auto is default, but we'll be explicit
                        )
                        for chunk in response:
                            # process normal response and tool_calls response
                            if len(chunk.choices) > 0:
                                deltas = chunk.choices[0].delta
                                if deltas is not None:
                                    if deltas.content is not None:
                                        full_response += deltas.content  # ["answer"]  # .choices[0].delta.get("content", "")
                                        time.sleep(0.001)
                                        message_placeholder.markdown(full_response + "▌")
                                    elif deltas.tool_calls is not None:
                                        if deltas.tool_calls[0].id is not None:
                                            if cur_func_call["id"] is not None:
                                                if cur_func_call["id"] != deltas.tool_calls[0].id:
                                                    tool_calls.append(cur_func_call)
                                                    cur_func_call = {"id": None, "type": "function",
                                                                     "function": {"arguments": "", "name": None}}
                                            cur_func_call["id"] = deltas.tool_calls[0].id
                                        if deltas.tool_calls[0].function.name is not None:
                                            cur_func_call["function"]["name"] = deltas.tool_calls[0].function.name
                                        if deltas.tool_calls[0].function.arguments is not None:
                                            cur_func_call["function"]["arguments"] += deltas.tool_calls[0].function.arguments
                                if chunk.choices[0].finish_reason == "tool_calls":
                                    tool_calls.append(cur_func_call)
                                    cur_func_call = {"name": None, "arguments": "", "id": None}
                                    # function call here using func_call
                                    # print("call tool here")
                                    response_message = {"role": "assistant", "content": None, "tool_calls": tool_calls}
                    except Exception as e:
                        print(f"Error found: {e}")
                        print(st.session_state["FreeChatMessages"])
                        st.error(f"请求失败了，请稍后再试。\n\n{str(e)}")
                        del st.session_state["FreeChatMessages"][model_len_before_input:]
                        st.stop()

                    # Step 2: check if the model wanted to call a function
                    if tool_calls:
                        # Step 3: call the function
                        # Note: the JSON response may not always be valid; be sure to handle errors
                        available_functions = {
                            "get_current_weather": get_current_weather,
                            # "create_img_by_dalle3": create_img_by_dalle3,
                            "create_img_from_siliconflow": create_img_from_siliconflow,
                        }  # only one function in this example, but you can have multiple
                        # extend conversation with assistant's reply
                        display_len_before_tool = len(st.session_state["FreeChatMessagesDisplay"])
                        st.session_state["FreeChatMessages"].append(response_message)
                        st.session_state["FreeChatMessagesDisplay"].append(response_message)
                        # Step 4: send the info for each function call and function response to the model
                        try:
                            for tool_call in tool_calls:
                                function_name = tool_call["function"]["name"]
                                function_response = execute_function_call(available_functions, tool_call)
                                st.session_state["FreeChatMessages"].append(
                                    {
                                        "tool_call_id": tool_call["id"],
                                        "role": "tool",
                                        "name": function_name,
                                        "content": function_response,
                                    }
                                )  # extend conversation with function response
                                st.session_state["FreeChatMessagesDisplay"].append(
                                    {
                                        "tool_call_id": tool_call["id"],
                                        "role": "tool",
                                        "name": function_name,
                                        "content": function_response,
                                    }
                                )  # extend conversation with function response
                            second_response = st.session_state["FreeChatChain"].chat.completions.create(
                                model=st.session_state["FreeChatSetting"]["model"],
                                messages=build_api_messages(prompt),
                                max_tokens=st.session_state["FreeChatSetting"]["max_tokens"],
                                # default max tokens is low so set higher
                                temperature=st.session_state["FreeChatSetting"]["temperature"],
                                stream=True,
                            )  # get a new response from the model where it can see the function response
                            for chunk in second_response:
                                if len(chunk.choices) > 0:
                                    deltas = chunk.choices[0].delta
                                    if deltas is not None:
                                        if deltas.content is not None:
                                            full_response += deltas.content  # ["answer"]  # .choices[0].delta.get("content", "")
                                            time.sleep(0.001)
                                            message_placeholder.markdown(full_response + "▌")
                        except Exception as e:
                            print(f"Error found: {e}")
                            st.error(f"请求失败了，请稍后再试。\n\n{str(e)}")
                            del st.session_state["FreeChatMessages"][model_len_before_input:]
                            del st.session_state["FreeChatMessagesDisplay"][display_len_before_tool:]
                            st.stop()
                    else:
                        full_response = full_response
                message_placeholder.markdown(full_response)
                if not full_response:
                    st.warning("模型没有返回内容，请重试。")
                    del st.session_state["FreeChatMessages"][model_len_before_input:]
                    st.stop()

                st.session_state["FreeChatMessages"].append({"role": "assistant", "content": [{"type": "text", "text": full_response}]})
                if function_response.startswith("https://"):
                    st.session_state["FreeChatMessagesDisplay"].append(
                        {"role": "assistant", "content": [{"type": "text", "text": full_response}], "image": function_response})
                    st.image(function_response)
                    # Save AI response to database
                    save_chat_to_db(st.session_state.current_topic_id, "assistant", "text", full_response, function_response)
                else:
                    st.session_state["FreeChatMessagesDisplay"].append(
                        {"role": "assistant", "content": [{"type": "text", "text": full_response}]})
                    # Save AI response to database
                    save_chat_to_db(st.session_state.current_topic_id, "assistant", "text", full_response)
                print("AI: " + full_response)
                if aa_voice_name != "None":
                    chatbot_azure.text_2_speech(full_response.replace("*","").replace("#", ""), aa_voice_name)
                btn_placeholder.button(label="Play", key="current", on_click=chatbot_azure.text_2_speech,
                                       args=(full_response.replace("*","").replace("#", ""), aa_voice_name,))


if __name__ == "__main__":
    main()
