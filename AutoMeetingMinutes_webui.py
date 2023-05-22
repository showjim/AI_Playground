import streamlit as st
from analyse_audio import (
    extract_subtitle,
    identify_speaker,
    output_subtitle
)
from generate_meeting_minutes import llm_chat
import pandas as pd
import json, os


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

work_path = st.text_area("Insert your working path")
uploaded_path = ""

if st.button("Submit", type="primary"):

    if video_path is not None:
        # save file
        uploaded_path = os.path.join(work_path + "/tempDir", video_path.name)
        with open(uploaded_path, mode="wb") as f:
            f.write(video_path.getvalue())
        segments, new_file = extract_subtitle(uploaded_path, aa_file_type, aa_lang)
        # embeddings = embedding_audio(new_file, segments)
        segments_speaker = identify_speaker(new_file, segments, aa_spk_num)
        output_subtitle(new_file, segments_speaker)

        # Query the agent.
        response = llm_chat(work_path + "/tempDir/output",
                            work_path + "/index",
                            work_path)
        st.text(response)
        ## Decode the response.
        # decoded_response = decode_response(response)
        ## Write the response to the Streamlit app.
        # write_response(decoded_response)

if st.button("Re-Generate", type="secondary"):
    # Query the agent.
    response = llm_chat(work_path + "/tempDir/output",
                        work_path + "/index",
                        work_path)
    st.text(response)