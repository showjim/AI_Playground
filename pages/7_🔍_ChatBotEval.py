import streamlit as st
import json, os, shutil, openai, csv
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from langchain.evaluation.qa import QAGenerateChain
from langchain.vectorstores import FAISS, Qdrant
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings import (
    OpenAIEmbeddings,
    HuggingFaceEmbeddings,
)
from langchain.document_loaders import (
    PyMuPDFLoader,
)
from langchain.retrievers import (
    SVMRetriever,
    AzureCognitiveSearchRetriever,
    TFIDFRetriever,
    ContextualCompressionRetriever,
)
from langchain.retrievers.document_compressors import LLMChainExtractor
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

        # Aure Cognitive Search
        os.environ["AZURE_COGNITIVE_SEARCH_SERVICE_NAME"] = config_details['AZURE_COGNITIVE_SEARCH_SERVICE_NAME']
        os.environ["AZURE_COGNITIVE_SEARCH_INDEX_NAME"] = config_details['AZURE_COGNITIVE_SEARCH_INDEX_NAME']
        os.environ["AZURE_COGNITIVE_SEARCH_API_KEY"] = os.getenv('AZURE_COGNITIVE_SEARCH_API_KEY')
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
    elif embedding_method == "HuggingFace":
        embeddings = HuggingFaceEmbeddings()
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
    if "EvalQAresults" not in st.session_state:
        st.session_state["EvalQAresults"] = None

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
        aa_vector_store = st.radio(label="`Choose vector store`",
                                   options=["FAISS", "Qdrant"],
                                   index=0,
                                   on_change=set_reload_setting_flag)
        aa_retriver = st.radio(label="`Choose retriever`",
                               options=["Similarity Search", "MMR", "Contextual Compression","Azure Cognitive Search", "SVM", "TFIDF"],
                               index=0,
                               on_change=set_reload_setting_flag)
        aa_chunk_num = st.select_slider("`Choose # chunks to retrieve`",
                                        options=[3, 4, 5, 6, 7, 8],
                                        on_change=set_reload_setting_flag)

        # 4. Embedding
        aa_embedding_method = st.radio(label="`Choose embeddings`",
                                       options=["OpenAI", "HuggingFace"],
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
                    # for i in range(len(uploaded_paths)):
                    uploaded_path = uploaded_paths[0]
                    texts = TextSplitter.split_documents(documents)
                    st.session_state["EvalTexts"] = texts

                    # search & retriver
                    # FAISS/Qdrant: save documents as index, and then load them(not use the save function)
                    # Others do not support save for now
                    if aa_vector_store == "FAISS":
                        single_index_name = "./index/" + Path(uploaded_path).stem + ".faiss"
                        if Path(single_index_name).is_file() == False:
                            tmpdb = FAISS.from_documents(texts, EmbeddingModel)
                            # tmpdocsearch = tmpdb.as_retriever(search_kwargs={"k": aa_chunk_num})
                            tmpdb.save_local("./index/", Path(uploaded_path).stem)
                        else:
                            tmpdb = FAISS.load_local("./index/", EmbeddingModel, Path(uploaded_path).stem)
                    elif aa_vector_store == "Qdrant":
                        single_index_name = "./index/" + Path(uploaded_path).stem + ".qdrant"
                        tmpdb = Qdrant.from_documents(texts,
                                                      EmbeddingModel,
                                                      path="./index/",
                                                      collection_name=Path(uploaded_path).stem
                                                      )

                    if aa_retriver == "Similarity Search":
                        # if Path(single_index_name).is_file() == False:
                        #     tmpdb = FAISS.from_documents(texts, EmbeddingModel)
                        #     # tmpdocsearch = tmpdb.as_retriever(search_kwargs={"k": aa_chunk_num})
                        #     tmpdb.save_local("./index/", Path(uploaded_path).stem)
                        # else:
                        #     tmpdb = FAISS.load_local("./index/", EmbeddingModel, Path(uploaded_path).stem)
                        tmpdocsearch = tmpdb.as_retriever(search_kwargs={"k": aa_chunk_num}) # default is "similarity"
                    elif aa_retriver == "MMR":
                        # like "Similarity Search" but fetch more splits and return a more diversity smaller splits
                        # if Path(single_index_name).is_file() == False:
                        #     tmpdb = FAISS.from_documents(texts, EmbeddingModel)
                        #     # tmpdocsearch = tmpdb.as_retriever(search_kwargs={"k": aa_chunk_num})
                        #     tmpdb.save_local("./index/", Path(uploaded_path).stem)
                        # else:
                        #     tmpdb = FAISS.load_local("./index/", EmbeddingModel, Path(uploaded_path).stem)
                        tmpdocsearch = tmpdb.as_retriever(search_type="mmr", search_kwargs={"k": aa_chunk_num}) # default fetch_k = 20
                    elif aa_retriver == "Contextual Compression":
                        # find the splits and compress them, then return only the relevant info
                        # if Path(single_index_name).is_file() == False:
                        #     tmpdb = FAISS.from_documents(texts, EmbeddingModel)
                        #     # tmpdocsearch = tmpdb.as_retriever(search_kwargs={"k": aa_chunk_num})
                        #     tmpdb.save_local("./index/", Path(uploaded_path).stem)
                        # else:
                        #     tmpdb = FAISS.load_local("./index/", EmbeddingModel, Path(uploaded_path).stem)
                        compressor = LLMChainExtractor.from_llm(LlmModel)
                        tmpdocsearch = ContextualCompressionRetriever(
                            base_compressor=compressor,
                            base_retriever=tmpdb.as_retriever(search_type="mmr")
                        )
                    elif aa_retriver == "SVM":
                        tmpdocsearch = SVMRetriever.from_documents(texts, EmbeddingModel, k=aa_chunk_num)
                    elif aa_retriver == "TFIDF":
                        tmpdocsearch = TFIDFRetriever.from_documents(texts, k=aa_chunk_num)
                    elif aa_retriver == "Azure Cognitive Search":
                        tmpdocsearch = AzureCognitiveSearchRetriever(content_key="content", top_k=aa_chunk_num)

                    docsearch = tmpdocsearch

                    # make chain
                    # qa_chain = RetrievalQA.from_chain_type(LlmModel, retriever=docsearch)
                    qa_chain = RetrievalQA.from_llm(llm=LlmModel, retriever=docsearch)
                    st.session_state["EvalQAChain"] = qa_chain

                    if len(uploaded_paths) > 0:
                        st.session_state["EvalUploadFile"] = Path(uploaded_path).stem
                        st.write(f"✅ " + ", ".join(uploaded_paths) + " uploaed")

        # Generate Q&A
        if st.button("Generate Q&A"):
            with st.spinner('Generating QnA pairs...'):
                texts = st.session_state["EvalTexts"]
                examples = []
                # # Hard - coded examples
                # examples = [
                #     {
                #         "query": "What did the president say about Ketanji Brown Jackson",
                #         "answer": "He praised her legal ability and said he nominated her for the supreme court.",
                #     },
                #     {"query": "What did the president say about Michael Jackson", "answer": "Nothing"},
                # ]

                example_gen_chain = QAGenerateChain.from_llm(LlmModel)
                new_examples = example_gen_chain.apply_and_parse([{"doc": t} for t in texts[:aa_eval_q]])
                # print(new_examples[0])

                # Combine examples
                examples += [tmp["qa_pairs"] for tmp in new_examples]
                st.session_state["EvalQAs"] = examples
        if st.session_state["EvalQAs"] is not None:
            df = show_csv(st.session_state["EvalQAs"])
            # save_csv(examples)
            # export srt file
            csv_QA = df.to_csv(index=False)
            tmpQAfile = st.session_state["EvalUploadFile"]
            st.download_button("Download .csv file", data=csv_QA, file_name=f"{tmpQAfile}_QA_autogen_pairs.csv")

    # upload QA pairs
    with qa_upload_container:
        # Upload QA file to EVAL
        file_paths = [st.file_uploader("2.Upload ground TRUE QAs", type=["csv"], accept_multiple_files=False)]
        if st.button("Upload QnA pairs", type="primary"):
            if file_paths is not None or len(file_paths) > 0:
                # load csv file to list of dict
                with st.spinner('Reading QnA pairs file'):
                    list_dicts = []
                    for file_path in file_paths:
                        upload_qa_df = pd.read_csv(file_path)
                        list_dicts += upload_qa_df.to_dict("records")
                    st.session_state["EvalUploadFile"] = Path(file_path.name).stem
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

                st.session_state["EvalQAresults"] = new_outputs

        if st.session_state["EvalQAresults"] is not None:
            df = show_csv(st.session_state["EvalQAresults"])
            # save_csv(new_outputs, "f1_grade_result.csv")
            csv_eval = df.to_csv()
            tmpEVALfile = st.session_state["EvalUploadFile"]
            st.download_button("Download EVAL file", data=csv_eval, file_name=f"{tmpEVALfile}_EVAL.csv")

if __name__ == "__main__":
    main()
