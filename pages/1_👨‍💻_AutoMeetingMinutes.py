import streamlit as st
from src.analyse_audio import (
    # extract_subtitle,
    extract_subtitle_api,
    identify_speaker,
    output_subtitle
)
import os
from src.chat import ChatBot
from src.ClsChatBot import ChatRobot
from pathlib import Path

# __version__ = "Beta V0.0.2"

# whisper
chatbot = ChatRobot()
chatbot.setup_env()
client_stt = chatbot.initial_whisper()

# LLM
work_path = os.path.abspath('.')
chat = ChatBot(work_path + "/tempDir/output",
                work_path + "/index",
                work_path)
chat.initial_llm("gpt-4o", 2048, 0.2)
query_str = """Please summary this meeting and output meeting minutes.
Please try to focus on the below requests, and use the bullet format to output the answers for each request: 
1. who attend the meeting?
2. Identify key decisions in the transcript.
3. What are the key action items in the meeting?
4. what are the next steps?
"""

st.title("👨‍💻Auto-Meeting-Minutes")

def main():
    # sidebar
    st.sidebar.expander("Settings")
    st.sidebar.subheader("Parameter for upload file")
    aa_lang = st.sidebar.selectbox("1.Language", ["en", "zh", "ja", "fr"])
    aa_file_type = st.sidebar.radio("2.File type", ["video", "audio"])
    aa_spk_num = st.sidebar.selectbox("3.Number of Speaker", list(range(1, 10)))
    aa_model_size = st.sidebar.selectbox("4.Whisper model", ["base","small","medium", "large-v1","large-v2"])

    # main page
    st.write("Please upload your video or audio below.")
    if "mediavectordb" not in st.session_state:
        st.session_state["mediavectordb"] = None

    if aa_file_type == "video":
        video_path = st.file_uploader("Upload a Video or Audio", type=["mp4","mkv","avi"])
    else:
        video_path = st.file_uploader("Upload a Video or Audio", type=["wav","mp3","m4a"])

    query_input = st.text_area("Insert your instruction", query_str)
    uploaded_path = ""

    if st.button("Submit", type="primary"):

        if video_path is not None:
            work_path = os.path.abspath('.')
            # save file
            uploaded_path = os.path.join(work_path + "/tempDir", video_path.name)
            with open(uploaded_path, mode="wb") as f:
                f.write(video_path.getvalue())
            # segments, new_file, srt_string, duration = extract_subtitle(uploaded_path, aa_file_type, aa_lang, aa_model_size)
            segments, new_file, srt_string, duration = extract_subtitle_api(uploaded_path, aa_file_type, client_stt, aa_lang, prompt="以下是普通话的句子。")

            # export srt file
            if srt_string != "":
                st.download_button("Download .srt file", data=srt_string, file_name=f"{video_path.name}.srt")

            # identify the speakers
            segments_speaker = identify_speaker(new_file, segments, aa_spk_num, duration)
            output_subtitle(new_file, segments_speaker)

            # Query the agent.
            with st.spinner('preparing answer'):
                st.session_state["mediavectordb"] = chat.setup_vectordb("./tempDir/output/" + Path(video_path.name).stem + ".txt")
                response = chat.chat(query_input, st.session_state["mediavectordb"])
            # response = llm_chat_langchain(query_input, work_path + "/tempDir/output",
            #                     work_path + "/index",
            #                     work_path)
            st.text(response)


    if st.button("Re-Generate", type="secondary"):
        # work_path = os.path.abspath('.')
        # Query the agent.
        # st.info('This is a purely informational message', icon="ℹ️")
        with st.spinner('preparing answer'):
            # doc_summary_index = chat.setup_vectordb(uploaded_path)
            response = chat.chat(query_input, st.session_state["mediavectordb"])
        # response = llm_chat_langchain(query_input, work_path + "/tempDir/output",
        #                             work_path + "/index",
        #                             work_path)
        st.markdown(response)
        # st.info('This is a purely informational message2', icon="ℹ️")

if __name__ == "__main__":
    main()