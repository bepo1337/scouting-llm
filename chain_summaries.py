import os
from typing import Dict, List

from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
import json
from pymilvus import Collection, connections
import model_definitions
from langchain_milvus import Milvus
from langchain_openai import AzureChatOpenAI

import prompt_templates

load_dotenv()
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
OPENAI_API_VERSION = os.getenv('OPENAI_API_VERSION')
model_name="gpt-4o"
llm = AzureChatOpenAI(openai_api_key=AZURE_OPENAI_API_KEY, deployment_name=model_name)

EMBEDDING_MODEL = "nomic-embed-text"
DIMENSIONS = 768
COLLECTION_NAME_SUMMARY = "structured_summary"
COLLECTION_NAME_SINGLE_REPORTS = "scouting"
VECTOR_STORE_URI = "http://localhost:19530"
OLLAMA_URI = "http://localhost:11434/v1"
COUNT_RETRIEVED_DOCUMENTS = 5

#Embedding
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

connection_args = {'uri': VECTOR_STORE_URI}
vectorstore_summaries = Milvus(
    embedding_function=embeddings,
    connection_args=connection_args,
    collection_name=COLLECTION_NAME_SUMMARY,
    vector_field="embeddings",
    primary_field="id",
    auto_id=True
)

vectorstore_reports = Milvus(
    embedding_function=embeddings,
    connection_args=connection_args,
    collection_name=COLLECTION_NAME_SINGLE_REPORTS,
    vector_field="embeddings",
    primary_field="id",
    auto_id=True
)

# Need this to do metadata search only
connections.connect("default", host="localhost", port="19530")
summary_collection = Collection(COLLECTION_NAME_SUMMARY)
summary_collection.load()

def format_docs_to_json(docs: [Document]) -> str:
    # for every player we want: player_id, summary
    listResponse = model_definitions.ListPlayerResponse(list=[])

    for doc in docs:
        playerID = doc.metadata['player_transfermarkt_id']
        summary = doc.page_content
        playerRes = model_definitions.PlayerIDWithSummaryAndFineGrainedReports(player_id=playerID,
                                                                               report_summary=summary,
                                                                                fine_grained_reports=[])
        listResponse.list.append(playerRes)

    return json.loads(listResponse.model_dump_json())

def get_vectorstore_results(vectorstore, query, position):
    filterExpression = get_position_filter_expr(position)
    documents = vectorstore.similarity_search(
        query=query,
        k=COUNT_RETRIEVED_DOCUMENTS,
        expr=filterExpression
    )
    return documents


def expand_query(query: str) -> str:
    prompt = prompt_templates.PROMPT_QUERY_INTO_STRUCTURED_QUERY_WITH_EXAMPLE + query
    answer = llm.invoke(prompt).content
    return answer

def rag_chain(query: str, position: str):
    expaneded_query = expand_query(query)
    documents = get_vectorstore_results(vectorstore_summaries, query, position)
    return format_docs_to_json(documents)


def get_position_filter_expr(position):
    milvus_position_key = model_definitions.get_position_key_from_value(position)
    filterExpression = f"main_position == '{milvus_position_key}'" if milvus_position_key is not None else None
    return filterExpression


def invoke_summary_chain(query: str, position: str) -> str:
    print(f"Summary chain: User query: {query}, position: {position}")
    return rag_chain(query, position)


def player_id_to_reports(documents: [Document]) -> Dict[int, List[str]]:
    return_object = {}
    for doc in documents:
        player_id = int(doc.metadata['player_transfermarkt_id'])
        if player_id in return_object:
            report_list = return_object[player_id]
            report_list.append(doc.page_content)
        else:
            return_object[player_id] = [doc.page_content]
    return return_object


def get_summary_for_player_id(player_id) -> str:
    response = summary_collection.query(expr=f"player_transfermarkt_id == '{player_id}'", output_fields=["text"])
    return response[0]['text']

def invoke_single_report_chain(query: str, position: str) -> str:
    print(f"Fine grained chain: User query: {query}, position: {position}")
    # Get 5 reports
    documents = get_vectorstore_results(vectorstore_reports, query, position)
    # Get unique ids
    id_to_reports = player_id_to_reports(documents)
    # Get summary for each unique player id
    listResponse = model_definitions.ListPlayerResponse(list=[])
    for id in id_to_reports:
        playerResponse = model_definitions.PlayerIDWithSummaryAndFineGrainedReports(player_id=id,
                                                                                    report_summary="",
                                                                                    fine_grained_reports=[])
        # get summary
        playerResponse.report_summary = get_summary_for_player_id(id)
        playerResponse.fine_grained_reports = id_to_reports[id]
        listResponse.list.append(playerResponse)


    return json.loads(listResponse.model_dump_json())


