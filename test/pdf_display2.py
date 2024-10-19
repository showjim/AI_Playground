import base64

import streamlit as st


uploaded = st.file_uploader(label="Please browse for a pdf file", type="pdf")
method = st.radio(label="Method", options=["Embed", "Iframe"], horizontal=True)
if uploaded is None:
    st.stop()

base64_pdf = base64.b64encode(uploaded.read()).decode("utf-8")
if method == "Embed":
    pdf_display = (
        f'<embed src="data:application/pdf;base64,{base64_pdf}" '
        'width="800" height="1000" type="application/pdf"></embed>'
    )
elif method == "Iframe":
    pdf_display = (
        f'<iframe src="data:application/pdf;base64,{base64_pdf}" '
        'width="800" height="1000" type="application/pdf"></iframe>'
    )
else:
    st.error(f"Unknown method: {method}")
    
st.markdown(pdf_display, unsafe_allow_html=True)
