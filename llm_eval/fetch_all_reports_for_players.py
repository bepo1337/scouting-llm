# Fetch all reports for players and save them to json file llm generated golden summary
import os

from dotenv import load_dotenv
from pymilvus import (
    connections,
    Collection,
)
from pydantic.v1 import BaseModel
from typing import List
from langchain_core.documents import Document
import json
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import AzureChatOpenAI

load_dotenv()
save_file_name = "data/summarization_with_golden_summaries_prod.json"
# expects the file_list_of_ids to be a json list of strings (TM IDs to be precise)
file_list_of_ids = "data/list_id_for_summaries_prod.json"

EMBEDDING_MODEL = "nomic-embed-text"
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
COUNT_RETRIEVED_DOCUMENTS = 50
COLLECTION_NAME = "scouting"
VECTOR_STORE_URI = "http://localhost:19530"
connection_args = {'uri': VECTOR_STORE_URI}

AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
OPENAI_API_VERSION = os.getenv('OPENAI_API_VERSION')


connections.connect("default", host="localhost", port="19530")
scouting_collection = Collection(COLLECTION_NAME)
scouting_collection.load()

llm = AzureChatOpenAI(openai_api_key=AZURE_OPENAI_API_KEY, deployment_name="gpt-4o")


class GoldenSummaryAndRetrievedDocuments(BaseModel):
    golden_summary: str
    retrieved_documents: List[Document]


class ListOfGoldenSummaryAndRetrievedDocuments(BaseModel):
    data: List[GoldenSummaryAndRetrievedDocuments]


with open(file_list_of_ids, 'r') as file:
    id_list = json.load(file)

print(id_list)

def get_llm_summary(documents) -> str:
    documentCount = 1
    report_string = ""
    for doc in documents:
        report_string += f"Report {documentCount}: {doc.page_content}\n\n"
        documentCount += 1

    messages = [
        (
            "system",
            "You are a helpful assistant in soccer scouting. Summarize the following reports about a player. Return only the summary and nothing else.",
        ),
        ("human", report_string),
    ]
    llm_summary = llm.invoke(messages).content
    return llm_summary

references_with_empty_summaries = ListOfGoldenSummaryAndRetrievedDocuments(data=[])
for player_id in id_list:
    # milvus filter
    expr = f"player_transfermarkt_id in ['{player_id}']"

    result = scouting_collection.query(expr=expr, output_fields=["scout_id", "player_transfermarkt_id", "text"])
    documents = []
    for res in result:
        if res['text'] != "":
            metadata = {"player_transfermarkt_id" : res['player_transfermarkt_id']}
            doc = Document(page_content=res['text'], metadata=metadata)

            documents.append(doc)

    llm_summary = get_llm_summary(documents)
    goldenSummaryAndRetrievedDocuments = GoldenSummaryAndRetrievedDocuments(golden_summary=llm_summary,
                                                                            retrieved_documents=documents)
    references_with_empty_summaries.data.append(goldenSummaryAndRetrievedDocuments)

with open(save_file_name, 'w') as file:
    print(f"writing to {save_file_name}...")
    file.write(references_with_empty_summaries.json())
    print(f"successfully writen to {save_file_name}")

