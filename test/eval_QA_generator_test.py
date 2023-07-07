import json, os, shutil, openai, csv
from dotenv import load_dotenv
from langchain.evaluation.qa import QAGenerateChain
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    PyMuPDFLoader,
)
from langchain.evaluation.qa import QAEvalChain, CotQAEvalChain
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import AzureOpenAI

from langchain.prompts.prompt import PromptTemplate
import pandas as pd
from evaluate import load, combine

def save_csv(examples, filename:str="generated_QA.csv"):
    df = pd.DataFrame(examples)
    df.to_csv(filename)
    # with open("generated_QA.csv","w") as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow("index", "question", "answer")

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
PROMPT = PromptTemplate(
    input_variables=["query", "result", "answer"], template=template
)

# Load OpenAI key
if os.path.exists(os.path.join("../", "key.txt")):
    shutil.copyfile(os.path.join("../", "key.txt"), "../.env")
    load_dotenv()
else:
    print("key.txt with OpenAI API is required")

# Load config values
if os.path.exists(os.path.join(r'../config.json')):
    with open(r'../config.json') as config_file:
        config_details = json.load(config_file)

    # Setting up the embedding model
    embedding_model_name = config_details['EMBEDDING_MODEL']
    openai.api_type = "azure"
    openai.api_base = config_details['OPENAI_API_BASE']
    openai.api_version = config_details['OPENAI_API_VERSION']
    openai.api_key = os.getenv("OPENAI_API_KEY")
else:
    print("config.json with Azure OpenAI config is required")

llm = AzureChatOpenAI(deployment_name=config_details['CHATGPT_MODEL'],
                               openai_api_key=openai.api_key,
                               openai_api_base=openai.api_base,
                               openai_api_type=openai.api_type,
                               openai_api_version=config_details['OPENAI_API_VERSION'],
                               max_tokens=1024,
                               temperature=0.2,
                              # model_kwargs={'engine': self.config_details['CHATGPT_MODEL']},
                               )

embeddings = OpenAIEmbeddings(deployment=config_details['EMBEDDING_MODEL'], chunk_size=1)

loader = PyMuPDFLoader("./An Introduction to Scan Test for Test Engineers_1.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
if 0:
    docsearch = FAISS.from_documents(texts, embeddings)
else:
    docsearch = FAISS.load_local("../index/", embeddings, "An Introduction to Scan Test for Test Engineers_1")

qa = RetrievalQA.from_llm(llm=llm, retriever=docsearch.as_retriever())

# Hard-coded examples
examples = [
    {
        "query": "What did the president say about Ketanji Brown Jackson",
        "answer": "He praised her legal ability and said he nominated her for the supreme court.",
    },
    {"query": "What did the president say about Michael Jackson", "answer": "Nothing"},
]

example_gen_chain = QAGenerateChain.from_llm(llm)
new_examples = example_gen_chain.apply_and_parse([{"doc": t} for t in texts[:5]])
print(new_examples[0])

# Combine examples
examples += new_examples
save_csv(examples)

# Start predict
predictions = qa.apply(examples)
llm2 = AzureChatOpenAI(deployment_name=config_details['CHATGPT_MODEL'],
                               openai_api_key=openai.api_key,
                               openai_api_base=openai.api_base,
                               openai_api_type=openai.api_type,
                               openai_api_version=config_details['OPENAI_API_VERSION'],
                               max_tokens=1024,
                               temperature=0,
                              # model_kwargs={'engine': self.config_details['CHATGPT_MODEL']},
                               )
eval_chain = QAEvalChain.from_llm(llm2, prompt=PROMPT)
# eval_chain = CotQAEvalChain.from_llm(llm2)
graded_outputs = eval_chain.evaluate(examples, predictions)
for i, eg in enumerate(examples[:3]):
    print(f"Example {i}:")
    print("Question: " + predictions[i]["query"])
    print("Real Answer: " + predictions[i]["answer"])
    print("Predicted Answer: " + predictions[i]["result"])
    print("Predicted Grade: " + graded_outputs[i]["text"])
    print()

# output with binary eval
outputs = [
            {
                "query": predictions[i]["query"],
                "answer": predictions[i]["answer"],
                "result": predictions[i]["result"],
                "grade": graded_outputs[i]["text"]
            }
            for i, example in enumerate(examples)
        ]
save_csv(outputs, "grade_result.csv")

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

if 1:
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
            "result": outputs[i]["result"],
            "grade": outputs[i]["grade"],
            "exact_match": results[i]["exact_match"],
            "f1": results[i]["f1"]
        }
        for i, example in enumerate(outputs)
    ]
    save_csv(new_outputs, "f1_grade_result.csv")
else:
    clf_metrics = combine(["accuracy", "f1", "precision", "recall"])
    results = []
    for i in range(len(new_examples)):
        results.append(clf_metrics.compute(
            references=[new_examples[i]],
            predictions=[predictions[i]],
        ))

    print(results)