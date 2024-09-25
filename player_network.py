import os

from pymilvus import connections, Collection

# Configuration for Milvus
MILVUS_HOST = os.getenv("MILVUS_HOST_NAME", "localhost")
MILVUS_PORT = "19530"
COLLECTION_NAME = "summary_reports"

# Establish connection to Milvus
connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
collection = Collection(COLLECTION_NAME)
collection.load()

def get_player_embedding(player_transfermarkt_id: str):
    # Retrieve the embedding for a player given their player_transfermarkt_id
    query = f"player_transfermarkt_id == '{player_transfermarkt_id}'"
    search_results = collection.query(query, output_fields=["embeddings"])
    if search_results:
        return search_results[0]["embeddings"]
    else:
        raise ValueError(f"No embeddings found for player Transfermarkt ID {player_transfermarkt_id}")

def get_similar_players(player_transfermarkt_id: str):
    query_embedding = get_player_embedding(player_transfermarkt_id)

    # Define how many similar players to return
    top_k = 10

    # Search for similar players in Milvus
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}  # Adjust nprobe based on your setup
    search_results = collection.search(
        data=[query_embedding],
        anns_field="embeddings",
        param=search_params,
        limit=top_k + 1,  # Add +1 to include the player itself in the results so we can filter it out
        output_fields=["player_transfermarkt_id"]
    )
    print("Search results by similarity Search:", search_results)
    similar_players = []
    for result in search_results[0]:
        found_player_transfermarkt_id = result.entity.get("player_transfermarkt_id")
        if found_player_transfermarkt_id != str(player_transfermarkt_id):  # Exclude the original player from results
            distance = result.distance
            similar_players.append({"player_transfermarkt_id": found_player_transfermarkt_id, "distance": distance})

    print(similar_players)

    return similar_players