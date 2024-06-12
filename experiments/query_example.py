import numpy as np
from pymilvus import Collection, connections, DataType, model

connections.connect("default", host="localhost", port="19530")
collection = Collection("scouting")
collection.load()

def create_embedding(text: str) -> DataType.FLOAT_VECTOR:
    embedding_function = model.DefaultEmbeddingFunction()
    return embedding_function([text])

def query_vector_similarity():
    # search based on vector similarity, is a list cuz we can pass multiple at once
    query_vectors = create_embedding("give messi now")
    search_params = {
        "metric_type": "L2", #euclidean distance for vector comparisons
        "params": {"nprobe": 10},  # number of clusters to search
    }

    result = collection.search(query_vectors, "embeddings", search_params, limit=2,
                                        output_fields=["id", "text", "player_transfermarkt_id"])

    for hits in result:
        for hit in hits:
            print(f"hit: {hit}, scouting report text field: {hit.entity.get('text')}")


def query_hybrid():
    query_vectors = create_embedding("give next messi now")
    search_params = {
        "metric_type": "L2", #euclidean distance for vector comparisons
        "params": {"nprobe": 10},  # number of clusters to search
    }

    result = collection.search(query_vectors, "embeddings", search_params, limit=2,
                                        expr="scout_id == '1234'", output_fields=["id", "text"])

    for hits in result:
        for hit in hits:
            print(f"hit: {hit}, text field: {hit.entity.get('text')}")


def query_metadata():
    result = collection.query(expr="grade_rating > 0.35", output_fields=["id", "text", "grade_rating"])
    print(f"query result:\n-{result}")


print("Query via vector similarity:\n")
query_vector_similarity()
print("\n\n")


print("Query via metadata searching (all reports with grade_rating > 0.35):\n")
query_metadata()
print("\n\n")

print("Query via hybrid search:\n")
query_hybrid()
