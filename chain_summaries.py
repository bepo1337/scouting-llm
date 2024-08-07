from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
import json
from collections import defaultdict

import model_definitions
# from langchain_community.vectorstores import Milvus
from langchain_milvus import Milvus

# MODEL = "mistral"
EMBEDDING_MODEL = "nomic-embed-text"
DIMENSIONS = 768
COLLECTION_NAME = "scouting_summary"
VECTOR_STORE_URI = "http://localhost:19530"
OLLAMA_URI = "http://localhost:11434/v1"
COUNT_RETRIEVED_DOCUMENTS = 5

#Embedding
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

connection_args = {'uri': VECTOR_STORE_URI}
vectorstore = Milvus(
    embedding_function=embeddings,
    connection_args=connection_args,
    collection_name=COLLECTION_NAME,
    vector_field="embeddings",
    primary_field="id",
    auto_id=True
)

# retriever = vectorstore.as_retriever(search_kwargs={'k': COUNT_RETRIEVED_DOCUMENTS})

def format_docs_to_json(docs: [Document]) -> str:
    # for every player we want: player_id, summary
    listResponse = model_definitions.ListPlayerResponse(list=[])

    for doc in docs:
        playerID = doc.metadata['player_transfermarkt_id']
        summary = doc.page_content
        playerRes = model_definitions.PlayerResponse(player_id=playerID, report_summary=summary)
        listResponse.list.append(playerRes)

    return json.loads(listResponse.model_dump_json())


def rag_chain(query: str, position: str):
    milvus_position_key = model_definitions.get_position_key_from_value(position)
    filterExpression = f"main_position == '{milvus_position_key}'" if milvus_position_key is not None else None
    documents = vectorstore.similarity_search(
        query=query,
        k=COUNT_RETRIEVED_DOCUMENTS,
        expr=filterExpression
        # expr=f"main_position == '{milvus_position_key}'"
    )
    return format_docs_to_json(documents)


def invoke_chain(query: str, position: str) -> str:
    print(f"User query: {query}, position: {position}")
    return rag_chain(query, position)

