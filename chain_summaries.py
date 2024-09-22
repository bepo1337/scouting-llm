import os
from typing import Dict, List

from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
import json
from pymilvus import Collection, connections
from model_definitions import ComparePlayerRequestPayload, ComparePlayerResponsePayload
import model_definitions
from langchain_milvus import Milvus
from langchain_openai import AzureChatOpenAI

import prompt_templates
from rdb_access import fetch_reports_from_rdbms, fetch_name_from_rdbms

load_dotenv()
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
OPENAI_API_VERSION = os.getenv('OPENAI_API_VERSION')
model_name = "gpt-4o"
llm = AzureChatOpenAI(openai_api_key=AZURE_OPENAI_API_KEY, deployment_name=model_name)

EMBEDDING_MODEL = "nomic-embed-text"
DIMENSIONS = 768
COLLECTION_NAME_SUMMARY = "summary_reports"
COLLECTION_NAME_SINGLE_REPORTS = "original_reports"
VECTOR_STORE_URI = os.getenv("VECTOR_STORE_URL", "http://localhost:19530")
OLLAMA_URI = os.getenv("OLLAMA_URI", "http://localhost:11434/v1")
COUNT_RETRIEVED_DOCUMENTS = 5

# Embedding
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
milvus_host_name = os.getenv("MILVUS_HOST_NAME", "localhost")
connections.connect("default", host=milvus_host_name, port="19530")
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
    print(expaneded_query)
    documents = get_vectorstore_results(vectorstore_summaries, expaneded_query, position)
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
    if len(response) == 0:
        return "PLAYER_NOT_IN_SUMMARY"

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


### Below this for comparing players. Using the same configured LLM

def getComparisonTopicString(comparePlayersPayload: ComparePlayerRequestPayload):
    # create list with the strings
    topic_list = []
    if comparePlayersPayload.offensive is True:
        topic_list.append("offensive")

    if comparePlayersPayload.defensive is True:
        topic_list.append("defensive")

    if comparePlayersPayload.strenghts is True:
        topic_list.append("strengths")

    if comparePlayersPayload.weaknesses is True:
        topic_list.append("weaknesses")

    single_string_topics = ";".join(topic_list)
    single_string_topics += ";" + comparePlayersPayload.otherText
    return single_string_topics


def format_reports(reports, name) -> str:
    formatted_report = ""
    for i, report in enumerate(reports, 1):
        formatted_report += f"{name} report {i}: {report}\n"
    formatted_report += "###"
    return formatted_report


def replace_compare_placeholders(comparePlayersPayload: ComparePlayerRequestPayload):
    player_left_id = comparePlayersPayload.player_left
    player_right_id = comparePlayersPayload.player_right
    # prompt template
    query = prompt_templates.PROMPT_COMPARE_PLAYERS_NO_EXAMPLE
    comparison_topic_string = getComparisonTopicString(comparePlayersPayload)
    query = query.replace("{COMPARISON_TOPICS}", comparison_topic_string)

    player_left_name = fetch_name_from_rdbms(player_left_id)
    player_right_name = fetch_name_from_rdbms(player_right_id)
    query = query.replace("{FIRST_PLAYER_NAME}", player_left_name)
    query = query.replace("{SECOND_PLAYER_NAME}", player_right_name)

    # Only get single reports if othertext is filled --> can compare results
    if comparePlayersPayload.otherText != "":
        # fetch single reports
        player_left_reports = fetch_reports_from_rdbms(comparePlayersPayload.player_left)
        player_right_reports = fetch_reports_from_rdbms(comparePlayersPayload.player_right)
        query = query.replace("{FIRST_PLAYER_SINGLE_REPORTS}", format_reports(player_left_reports, player_left_name))
        query = query.replace("{SECOND_PLAYER_SINGLE_REPORTS}", format_reports(player_right_reports, player_right_name))

    player_left_summary = get_summary_for_player_id(player_left_id)
    player_right_summary = get_summary_for_player_id(player_right_id)
    query = query.replace("{FIRST_PLAYER_SUMMARY}", player_left_summary)
    query = query.replace("{SECOND_PLAYER_SUMMARY}", player_right_summary)

    return query


def llm_compare_players(comparePlayersPayload: ComparePlayerRequestPayload):
    query = replace_compare_placeholders(comparePlayersPayload)
    # prompt llm
    comparison = llm.invoke(query).content
    # parse to response object
    player_left_name = fetch_name_from_rdbms(comparePlayersPayload.player_left)
    player_right_name = fetch_name_from_rdbms(comparePlayersPayload.player_right)
    responsePayload = ComparePlayerResponsePayload(player_left=comparePlayersPayload.player_left,
                                                   player_left_name=player_left_name,
                                                   player_right=comparePlayersPayload.player_right,
                                                   player_right_name=player_right_name,
                                                   comparison=comparison)
    return responsePayload
