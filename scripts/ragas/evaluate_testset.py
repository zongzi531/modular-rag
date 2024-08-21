import requests
import json
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from datasets import load_from_disk
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas import evaluate
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# 自己的 RAG 流程，请自行替换
def generate_answer_by_rag(questions):
  answers = []
  t = tqdm(total=len(questions), desc="RAG Process")
  for question in questions:
    data = {
      "messages": [{"role": "user", "content": question}],
      "appId": os.environ.get("RAG_APP_ID"),
      "stream": False
    }
    headers = { "Content-type": "application/json", "Cookie": os.environ.get("RAG_COOKIE")}
    res = requests.post(os.environ.get("RAG_REQUEST_URL"), data=json.dumps(data), headers=headers)
    try:
      answer = res.json()["choices"][0]["message"]["content"]
      answers.append(answer)
    except:
      answers.append("none")
    t.update(1)
    
  return answers

testset = load_from_disk("./testset")
answers = generate_answer_by_rag(testset["question"])
testset = testset.add_column("answer", answers)

llm = ChatOpenAI(model="qwen2:72b-instruct")
embeddings = OpenAIEmbeddings(model="bge-large-zh-v1.5", check_embedding_ctx_length=False, show_progress_bar=True)

result = evaluate(
    testset,
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
    ],
    llm=llm,
    embeddings=embeddings,
)

result.to_pandas().to_csv("./result.csv")
