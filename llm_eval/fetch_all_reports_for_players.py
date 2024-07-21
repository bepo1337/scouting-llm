# Fetch all reports for players and save them to json file with empty golden summary
import json

from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import OllamaEmbeddings

save_file_name = "data/summarization_without_golden_summaries_prod.json"
file_list_of_ids = "data/list_id_for_summaries_prod.json"

EMBEDDING_MODEL = "nomic-embed-text"
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
COUNT_RETRIEVED_DOCUMENTS = 50
COLLECTION_NAME = "scouting"
VECTOR_STORE_URI = "http://localhost:19530"
connection_args = {'uri': VECTOR_STORE_URI}

connections.connect("default", host="localhost", port="19530")
scouting_collection = Collection(COLLECTION_NAME)
scouting_collection.load()



# vectorstore = Milvus(
#     embedding_function=embeddings,
#     connection_args=connection_args,
#     collection_name=COLLECTION_NAME,
#     vector_field="embeddings",
#     primary_field="id",
#     auto_id=True
# )

with open(file_list_of_ids, 'r') as file:
    id_list = json.load(file)

print(id_list)


for player_id in id_list:
    # Define the metadata filters
    expr = f"player_transfermarkt_id in ['{player_id}']"


    search_kwargs = {
        'k': COUNT_RETRIEVED_DOCUMENTS,
        'expr': expr
    }

    result = scouting_collection.query(expr=expr, output_fields=["scout_id", "player_transfermarkt_id", "text"])
    print(result)
