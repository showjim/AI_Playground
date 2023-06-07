import streamlit as st
import os
from src.chat import ChatBot


work_path = os.path.abspath('.')
chat = ChatBot(work_path + "/tempDir/output",
                work_path + "/index",
                work_path)
chat.initial_llm()

st.title("㉄ Chat with files")

def main():
    work_path = os.path.abspath('.')
    # main page
    st.write("Please upload your video or audio below.")

    file_path = st.file_uploader("Upload a document file", type=["pdf","txt","pptx","docx","html"])

    query_input = st.text_area("Insert your instruction")
    uploaded_path = ""

    # Generate empty lists for generated and past.
    ## generated stores AI generated responses
    if 'answers' not in st.session_state:
        st.session_state['answers'] = []
    ## past stores User's questions
    if 'questions' not in st.session_state:
        st.session_state['questions'] = []

    if st.button("Submit", type="primary"):

        if file_path is not None:
            # work_path = os.path.abspath('.')
            # save file
            uploaded_path = os.path.join(work_path + "/tempDir/output", file_path.name)
            with open(uploaded_path, mode="wb") as f:
                f.write(file_path.getvalue())

            chat.setup_vectordb()
            ## generated stores langchain chain
            if "QA_chain" not in st.session_state:
                qa_chain = chat.chat_QA_langchain()
                st.session_state["QA_chain"] = qa_chain

            # Query the agent.
            with st.spinner('preparing answer'):
                response = st.session_state["QA_chain"]({"question": query_input})
            resp = response["answer"]
            st.session_state.questions.append(query_input)
            st.session_state.answers.append(resp)
    response_container = st.container()
    with response_container:
        if st.session_state['answers']:
            n = len(st.session_state['answers'])
            for i in range(n):
                st.markdown('-----------------')
                st.markdown('### ' + str(n-i-1) + '. ' + st.session_state['questions'][n-i-1])
                st.markdown(st.session_state["answers"][n-i-1])


if __name__ == "__main__":
    main()