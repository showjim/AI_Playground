import streamlit as st
import fitz

with st.sidebar:
    original_doc = st.file_uploader(
        "Upload PDF", accept_multiple_files=False, type="pdf"
    )
    text_lookup = st.text_input("Look for", max_chars=50)

if original_doc:
    with fitz.open(stream=original_doc.getvalue()) as doc:
        page_number = st.sidebar.number_input(
            "Page number", min_value=1, max_value=doc.page_count, value=1, step=1
        )
        page = doc.load_page(page_number - 1)

        if text_lookup:
            areas = page.search_for(text_lookup)

            for area in areas:
                page.add_rect_annot(area)

            pix = page.get_pixmap(dpi=120).tobytes()
            st.image(pix, use_column_width=True)