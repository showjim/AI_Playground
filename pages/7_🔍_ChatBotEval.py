import streamlit as st
import json, os, shutil, openai, csv
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from langchain.evaluation.qa import QAGenerateChain
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import (
    PyMuPDFLoader,
)
from langchain.retrievers import (
    SVMRetriever,
    AzureCognitiveSearchRetriever,
    TFIDFRetriever,
)
from langchain.prompts.prompt import PromptTemplate
import pandas as pd
from langchain.evaluation.qa import QAEvalChain, CotQAEvalChain
from evaluate import load, combine

work_path = os.path.abspath('.')

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
    else:
        print("config.json with Azure OpenAI config is required")


def set_reload_setting_flag():
    # st.write("New document need upload")
    st.session_state["evalreloadflag"] = True


def define_llm(model: str):
    if "gpt-35-turbo" in model:
        llm = AzureChatOpenAI(deployment_name=model,
                              openai_api_key=openai.api_key,
                              openai_api_base=openai.api_base,
                              openai_api_type=openai.api_type,
                              openai_api_version=openai.api_version,
                              max_tokens=1024,
                              temperature=0.2,
                              # model_kwargs={'engine': self.config_details['CHATGPT_MODEL']},
                              )
    elif "" in model:
        pass
    return llm


def define_retriver(retriver: str):
    if retriver == "Similarity Search":
        pass
    elif retriver == "Azure Cognitive Search":
        pass
    elif retriver == "SVM":
        pass
    return retriver


def define_splitter(splitter: str, chunk_size, chunk_overlap):
    text_splitter = None
    if splitter == "RecursiveTextSplitter":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif splitter == "CharacterTextSplitter":
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter


def define_embedding(embedding_method: str):
    embeddings = None
    if embedding_method == "OpenAI":
        embeddings = OpenAIEmbeddings(deployment="text-embedding-ada-002",
                                      model="text-embedding-ada-002",
                                      openai_api_base=openai.api_base,
                                      openai_api_type=openai.api_type,
                                      chunk_size=1,)
    elif embedding_method == "Azure Cognitive Search":
        pass
    return embeddings


def load_single_document(file_path):
    loader = PyMuPDFLoader(file_path)
    return loader.load()  # [0]


def save_csv(examples, filename:str="generated_QA.csv"):
    df = pd.DataFrame(examples)
    df.to_csv(filename)

def show_csv(examples):
    df = pd.DataFrame(examples)
    st.dataframe(df)
    return df

def main():
    # Initial
    setup_env()
    if "evalreloadflag" not in st.session_state:
        st.session_state["evalreloadflag"] = True
    if "EvalTexts" not in st.session_state:
        st.session_state["EvalTexts"] = None
    if "EvalQAs" not in st.session_state:
        st.session_state["EvalQAs"] = None
    if "EvalQAChain" not in st.session_state:
        st.session_state["EvalQAChain"] = None
    if "EvalUploadFile" not in st.session_state:
        st.session_state["EvalUploadFile"] = None

    # Setup Side Bar
    with st.sidebar:
        # 1. Model
        aa_llm_model = st.radio(label="`LLM Model`",
                                options=["gpt-35-turbo", "gpt-35-turbo-16k"],
                                index=0,
                                on_change=set_reload_setting_flag)
        # 2. Split
        aa_eval_q = st.slider(label="`Number of eval questions`",
                              min_value=1,
                              max_value=10,
                              value=5,
                              on_change=set_reload_setting_flag)
        aa_chunk_size = st.slider(label="`Choose chunk size for splitting`",
                                  min_value=500,
                                  max_value=2000,
                                  value=1000,
                                  on_change=set_reload_setting_flag)
        aa_overlap_size = st.slider(label="`Choose overlap for splitting`",
                                    min_value=0,
                                    max_value=200,
                                    value=0,
                                    on_change=set_reload_setting_flag)
        aa_split_methods = st.radio(label="`Split method`",
                                    options=["RecursiveTextSplitter", "CharacterTextSplitter"],
                                    index=0,
                                    on_change=set_reload_setting_flag)

        # 3. Retriver
        aa_retriver = st.radio(label="`Choose retriever`",
                               options=["Similarity Search", "Azure Cognitive Search", "SVM", "TFIDF"],
                               index=0,
                               on_change=set_reload_setting_flag)
        aa_chunk_num = st.select_slider("`Choose # chunks to retrieve`",
                                        options=[3, 4, 5, 6, 7, 8],
                                        on_change=set_reload_setting_flag)

        # 4. Embedding
        aa_embedding_method = st.radio(label="`Choose embeddings`",
                                       options=["OpenAI", "Azure Cognitive Search"],
                                       index=0,
                                       on_change=set_reload_setting_flag)

        # if st.session_state["evalreloadflag"] == True:
        LlmModel = define_llm(aa_llm_model)
        EmbeddingModel = define_embedding(aa_embedding_method)
        TextSplitter = define_splitter(aa_split_methods, aa_chunk_size, aa_overlap_size)
        st.session_state["evalreloadflag"] = False

    ## Main
    st.header("`Demo auto-evaluator`")

    # Prompt for evaluation
    template = """You are a teacher grading a quiz.
    You are given a question, the student's answer, and the true answer, and are asked to score the student answer as either CORRECT or INCORRECT.
    Write out in a step by step manner your reasoning to be sure that your conclusion is correct. Avoid simply stating the correct answer at the outset.

    Example Format:
    QUESTION: question here
    TRUE ANSWER: true answer here
    STUDENT ANSWER: student's answer here
    GRADE: CORRECT or INCORRECT here

    Grade the student answers based ONLY on their factual accuracy. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements. Begin! 

    QUESTION: {query}
    TRUE ANSWER: {answer}
    STUDENT ANSWER: {result}
    GRADE:"""
    input = st.text_area(label="Evaluation Prompt", value=template, )
    PROMPT = PromptTemplate(
        input_variables=["query", "answer", "result"], template=input
    )


    qa_gen_container = st.container()
    # st.write("2. Upload QA pairs")
    qa_upload_container = st.container()
    st.write("3. Run EVAL")
    eval_container = st.container()

    with qa_gen_container:
        # Upload file to generate Q&A pairs
        file_paths = [st.file_uploader("1.Upload document files to generate QAs",
                                      type=["pdf"],
                                      accept_multiple_files=False)]

        if st.button("Upload Ref Document", type="primary"):
            if file_paths is not None or len(file_paths) > 0:
                # save file
                with st.spinner('Reading file'):
                    uploaded_paths = []
                    for file_path in file_paths:
                        uploaded_paths.append(os.path.join(work_path + "/tempDir/output", file_path.name))
                        uploaded_path = uploaded_paths[-1]
                        with open(uploaded_path, mode="wb") as f:
                            f.write(file_path.getbuffer())

                # process file
                with st.spinner('Create vector DB'):
                    # load documents
                    documents = []
                    with tqdm(total=len(uploaded_paths), desc='Loading new documents', ncols=80) as pbar:
                        for uploaded_path in uploaded_paths:
                            documents = documents + load_single_document(uploaded_path)
                            pbar.update()

                    # split documents
                    for i in range(len(uploaded_paths)):
                        uploaded_path = uploaded_paths[i]
                        texts = TextSplitter.split_documents(documents)
                        st.session_state["EvalTexts"] = texts

                        # search & retriver
                        # FAISS: save documents as index, and then load them(not use the save function)
                        # Others do not support save for now
                        single_index_name = "./index/" + Path(uploaded_path).stem + ".faiss"
                        if Path(single_index_name).is_file() == False:
                            if aa_retriver == "Similarity Search":
                                tmpdocsearch = FAISS.from_documents(texts, EmbeddingModel)
                                tmpdocsearch.save_local("./index/", Path(uploaded_path).stem)
                            elif aa_retriver == "SVM":
                                tmpdocsearch = SVMRetriever.from_documents(texts, EmbeddingModel)
                            elif aa_retriver == "TFIDF":
                                tmpdocsearch = TFIDFRetriever.from_documents(texts)
                            elif aa_retriver == "Azure Cognitive Search":
                                tmpdocsearch = AzureCognitiveSearchRetriever(content_key="content", top_k=aa_chunk_num)

                            if i == 0:
                                docsearch = tmpdocsearch
                            else:
                                # not used
                                docsearch.merge_from(tmpdocsearch)
                        else:
                            # only used for FAISS
                            if i == 0:
                                if aa_retriver == "Similarity Search":
                                    docsearch = FAISS.load_local("./index/", EmbeddingModel, Path(uploaded_path).stem)
                            else:
                                # not used
                                docsearch.merge_from(FAISS.load_local("./index/", EmbeddingModel, Path(uploaded_path).stem))

                    # make chain
                    # qa_chain = RetrievalQA.from_chain_type(LlmModel, retriever=docsearch)
                    qa_chain = RetrievalQA.from_llm(llm=LlmModel, retriever=docsearch.as_retriever())
                    st.session_state["EvalQAChain"] = qa_chain

                    if len(uploaded_paths) > 0:
                        st.session_state["EvalUploadFile"] = Path(uploaded_path).stem
                        st.write(f"✅ " + ", ".join(uploaded_paths) + " uploaed")

        # Generate Q&A
        if st.button("Generate Q&A"):
            with st.spinner('Generating QnA pairs...'):
                texts = st.session_state["EvalTexts"]
                # Hard - coded examples
                examples = [
                    {
                        "query": "What did the president say about Ketanji Brown Jackson",
                        "answer": "He praised her legal ability and said he nominated her for the supreme court.",
                    },
                    {"query": "What did the president say about Michael Jackson", "answer": "Nothing"},
                ]

                example_gen_chain = QAGenerateChain.from_llm(LlmModel)
                new_examples = example_gen_chain.apply_and_parse([{"doc": t} for t in texts[:aa_eval_q]])
                # print(new_examples[0])

                # Combine examples
                examples += [tmp["qa_pairs"] for tmp in new_examples]
                st.session_state["EvalQAs"] = examples
                df = show_csv(examples)
                # save_csv(examples)
                # export srt file
                csv = df.to_csv(index=False)
                tmpfile = st.session_state["EvalUploadFile"]
                st.download_button("Download .csv file", data=csv, file_name=f"{tmpfile}_QA_autogen_pairs.csv")

    # upload QA pairs
    with qa_upload_container:
        # Upload QA file to EVAL
        file_paths = [st.file_uploader("2.Upload ground TRUE QAs",
                                       type=["csv"],
                                       accept_multiple_files=False)]
        if st.button("Upload QnA pairs", type="primary"):
            if file_paths is not None or len(file_paths) > 0:
                # save file
                with st.spinner('Reading QnA pairs file'):
                    for file_path in file_paths:
                        upload_qa_df = pd.read_csv(file_path)
                        list_dicts = upload_qa_df.to_dict("records")
                        st.session_state["EvalQAs"] = list_dicts
                        # print(st.session_state["EvalQAs"])

    with eval_container:
        # Start EVAL
        if st.button("Start EVAL", type="primary"):
            with st.spinner('Evaluating the LLM setting...'):
                if st.session_state["EvalQAs"] is not None:
                    examples = st.session_state["EvalQAs"]
                else:
                    st.toast("No QnA pairs are uploaded or generated!")
                if st.session_state["EvalQAChain"] is not None:
                    qa_chain = st.session_state["EvalQAChain"]
                else:
                    st.toast("No qa chain is created!")

                predictions = qa_chain.apply(examples)
                eval_chain = QAEvalChain.from_llm(LlmModel, prompt=PROMPT)
                # eval_chain = CotQAEvalChain.from_llm(llm2)
                graded_outputs = eval_chain.evaluate(examples, predictions)
                # for i, eg in enumerate(examples[:3]):
                #     print(f"Example {i}:")
                #     print("Question: " + predictions[i]["query"])
                #     print("Real Answer: " + predictions[i]["answer"])
                #     print("Predicted Answer: " + predictions[i]["result"])
                #     print("Predicted Grade: " + graded_outputs[i]["results"])
                #     print()

                # output with binary eval
                outputs = [
                    {
                        "query": predictions[i]["query"],
                        "answer": predictions[i]["answer"],
                        "predict result": predictions[i]["result"],
                        "grade": graded_outputs[i]["results"]
                    }
                    for i, example in enumerate(examples)
                ]
                # show_csv(outputs)
                # save_csv(outputs, "grade_result.csv")

                # perform SQUaD Eval
                # Some data munging to get the examples in the right format
                for i, eg in enumerate(examples):
                    eg["id"] = str(i)
                    eg["answers"] = {"text": [eg["answer"]], "answer_start": [0]}
                    predictions[i]["id"] = str(i)
                    predictions[i]["prediction_text"] = predictions[i]["result"]

                for p in predictions:
                    del p["result"]
                    del p["query"]
                    del p["answer"]
                    # del p["text"]

                new_examples = examples.copy()
                for eg in new_examples:
                    del eg["query"]
                    del eg["answer"]

                squad_metric = load("squad")
                results = []
                for i in range(len(new_examples)):
                    results.append(squad_metric.compute(
                        references=[new_examples[i]],
                        predictions=[predictions[i]],
                    ))

                print(results)

                new_outputs = [
                    {
                        "query": outputs[i]["query"],
                        "answer": outputs[i]["answer"],
                        "predict result": outputs[i]["predict result"],
                        "grade": outputs[i]["grade"],
                        "exact_match": results[i]["exact_match"],
                        "f1": results[i]["f1"]
                    }
                    for i, example in enumerate(outputs)
                ]
                df = show_csv(new_outputs)
                # save_csv(new_outputs, "f1_grade_result.csv")
                csv = df.to_csv()
                tmpfile = st.session_state["EvalUploadFile"]
                st.download_button("Download EVAL file", data=csv, file_name=f"{tmpfile}_EVAL.csv")

if __name__ == "__main__":
    main()
