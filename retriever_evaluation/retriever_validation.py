from pymilvus import connections, Collection
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import OllamaEmbeddings

EMBEDDING_MODEL = "nomic-embed-text"
DIMENSIONS = 768
COLLECTION_NAME = "scouting"
VECTOR_STORE_URI = "http://localhost:19530"

# Example requests and their validation sets
requests = {
    "request_1": ("I am looking for an attacking midfielder with excellent ball control and strong passing skills", [9, 35, 36, 37, 39, 49, 52, 59, 60, 61, 77, 81, 82, 85, 86, 89, 98, 99, 103, 116, 140, 153, 154, 158, 159, 171, 174, 178, 193, 194, 232, 245, 249, 250, 256, 259, 279]),
    "request_2": ("Robust central defender with good heading and tackling ability", [13, 18, 21, 40, 42, 46, 48, 57, 63, 64, 73, 74, 76, 95, 96, 111, 113, 115, 121, 122, 173, 196, 198, 221, 244, 275, 276, 292, 293, 294]),
    "request_3": ("I am looking for a fast winger with high dribbling power and precise crosses who can play on both the right and left side.", [3, 5, 12, 65, 66, 81, 82, 87, 89, 92, 112, 124, 132, 133, 134, 135, 136, 137, 139, 140, 170, 180, 181, 203, 205, 211, 212, 216, 217, 218, 219, 220, 296]),
    "request_4": ("I need a defensive midfielder with a high willingness to run and good tackling skills", [1, 2, 5, 7, 25, 60, 63, 71, 72, 73, 110, 116, 142, 145, 183, 184, 185, 196, 234, 235, 245, 257, 269, 270, 272, 287]),
    "request_5": ("Goalkeeper with quick reflexes and good penalty area control", [31, 78])
}

# Connect to Milvus
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

retriever = vectorstore.as_retriever(search_kwargs={'k': 2})

# Function to query Milvus
def query_milvus(request_text):
    # Search for similar embeddings in Milvus
    results = retriever.invoke(request_text)
    
    # Extract player_ids from results
    player_ids = [result.metadata['id'] for result in results]
    return player_ids


def calculate_metrics(request_text, validation_list):
    # Get the response (player list) from Milvus
    response_ids = query_milvus(request_text)
    
    # Obtain the set of all possible player IDs 
    all_player_ids = range(1,301)  #comprehensive set of all player IDs in your database
    
    # Create binary lists for the true and predicted values
    y_true = [1 if player_id in validation_list else 0 for player_id in all_player_ids]
    y_pred = [1 if player_id in response_ids else 0 for player_id in all_player_ids]
    
    # Calculate Precision, Recall, and F1-Score
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred, zero_division=1)
    
    return precision, recall, f1

# Loop through requests and calculate metrics
for req_name, (req_text, req_validation) in requests.items():
    precision, recall, f1 = calculate_metrics(req_text, req_validation)
    print(f"{req_name} - Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")
