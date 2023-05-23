import streamlit as st
from src.analyse_audio import (
    extract_subtitle,
    identify_speaker,
    output_subtitle
)
from src.generate_meeting_minutes import llm_chat, llm_chat_langchain
import os

query_str = """You are a helpful assistant to do meeting record.
Please summary this meeting record.
Please try to focus on the below requests, and use the bullet format to output the answers for each request: 
1. who attend the meeting?
2. Identify key decisions in the transcript.
3. What are the key action items in the meeting?
4. what are the next steps?
"""

st.title("👨‍💻Auto-Meeting-Minutes")

# sidebar
st.sidebar.expander("Settings")
st.sidebar.subheader("Parameter for upload file")
aa_lang = st.sidebar.selectbox("1.Language", ["en", "zh", "ja", "fr"])
aa_file_type = st.sidebar.radio("2.File type", ["video", "audio"])
aa_spk_num = st.sidebar.selectbox("3.Number of Speaker", list(range(1, 10)))

# main page
st.write("Please upload your video or audio below.")
if aa_file_type == "video":
    video_path = st.file_uploader("Upload a Video or Audio")
else:
    video_path = st.file_uploader("Upload a Video or Audio", type=["wav","mp3"])

query_input = st.text_area("Insert your working path", query_str)
uploaded_path = ""

if st.button("Submit", type="primary"):

    if video_path is not None:
        work_path = os.path.abspath('.')
        # save file
        uploaded_path = os.path.join(work_path + "/tempDir", video_path.name)
        with open(uploaded_path, mode="wb") as f:
            f.write(video_path.getvalue())
        segments, new_file = extract_subtitle(uploaded_path, aa_file_type, aa_lang)
        # embeddings = embedding_audio(new_file, segments)
        segments_speaker = identify_speaker(new_file, segments, aa_spk_num)
        output_subtitle(new_file, segments_speaker)

        # Query the agent.
        response = llm_chat(query_input, work_path + "/tempDir/output",
                            work_path + "/index",
                            work_path)
        st.text(response)


if st.button("Re-Generate", type="secondary"):
    work_path = os.path.abspath('.')
    # Query the agent.
    # st.info('This is a purely informational message', icon="ℹ️")
    response = llm_chat_langchain(query_input, work_path + "/tempDir/output",
                                work_path + "/index",
                                work_path)
    st.text(response)
    # st.info('This is a purely informational message2', icon="ℹ️")