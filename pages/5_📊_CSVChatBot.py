import streamlit as st
import os
from src.chat import AgentChatBot


work_path = os.path.abspath('.')
reset_db = False

st.title("📊 Chat with CSV files")


def main():
    work_path = os.path.abspath('.')
    # Layout of output/setup containers
    setup_container = st.container()
    instruction_container = st.container()
    response_container = st.container()

    if "csv_agent" not in st.session_state:
        st.session_state["csv_agent"] = AgentChatBot(work_path, work_path + "/tempDir/output")

    # main page
    with setup_container:
        st.write("Please upload your csv file below.")
        file_path = st.file_uploader("Upload a csv file", type=["csv"])
        if st.button("Upload"):
            if file_path is not None:
                # save file
                with st.spinner('Reading file'):
                    uploaded_path = os.path.join(work_path + "/tempDir/output", file_path.name)
                    with open(uploaded_path, mode="wb") as f:
                        f.write(file_path.getvalue())
                with st.spinner('Create CSV agent'):
                    st.session_state["csv_agent"].initial_llm("csv", uploaded_path)
                    # st.session_state["vectordb"] = chat.setup_vectordb(uploaded_path)
                st.write(f"✅ {file_path.name} ")

    with instruction_container:
        query_input = st.text_area("Insert your instruction")
        uploaded_path = ""

        # Generate empty lists for generated and past.
        ## generated stores AI generated responses
        if 'csv_agent_answers' not in st.session_state:
            st.session_state['csv_agent_answers'] = []
        ## past stores User's questions
        if 'csv_agent_questions' not in st.session_state:
            st.session_state['csv_agent_questions'] = []

        if st.button("Submit", type="primary"):

            if file_path is not None:
                # Query the agent.
                with st.spinner('preparing answer'):
                    response = st.session_state["csv_agent"].chat_csv_agent(query_input)

                resp = response
                st.session_state["csv_agent_questions"].append(query_input)
                st.session_state["csv_agent_answers"].append(resp)

    with response_container:
        if st.session_state['csv_agent_answers']:
            n = len(st.session_state['csv_agent_answers'])
            for i in range(n):
                st.markdown('-----------------')
                st.markdown('### ' + str(n-i-1) + '. ' + st.session_state['csv_agent_questions'][n-i-1])
                st.markdown(st.session_state["csv_agent_answers"][n-i-1])


if __name__ == "__main__":
    main()