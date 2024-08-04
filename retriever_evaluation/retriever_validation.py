from pymilvus import connections, Collection
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
import json
from rank_bm25 import BM25Okapi

EMBEDDING_MODEL = "nomic-embed-text"
DIMENSIONS = 768
COLLECTION_NAME = "testscouting"
VECTOR_STORE_URI = "http://localhost:19530"

model_name = 'nomic-ai/nomic-embed-text-v1'
model_kwargs = {'device': 'mps', 'trust_remote_code': True}
encode_kwargs = {'normalize_embeddings': False} 

# Example requests and their validation sets
requests = {
    "request_1": ("I am looking for an attacking midfielder with excellent ball control and strong passing skills", [9, 35, 36, 37, 39, 49, 52, 59, 60, 61, 77, 81, 82, 85, 86, 89, 98, 99, 103, 116, 140, 153, 154, 158, 159, 171, 174, 178, 193, 194, 232, 245, 249, 250, 256, 259, 279]),
    "request_2": ("Robust central defender with good heading and tackling ability", [13, 18, 21, 40, 42, 46, 48, 57, 63, 64, 73, 74, 76, 95, 96, 111, 113, 115, 121, 122, 173, 196, 198, 221, 244, 275, 276, 292, 293, 294]),
    "request_3": ("I am looking for a fast winger with high dribbling power and precise crosses who can play on both the right and left side.", [3, 5, 12, 65, 66, 81, 82, 87, 89, 92, 112, 124, 132, 133, 134, 135, 136, 137, 139, 140, 170, 180, 181, 203, 205, 211, 212, 216, 217, 218, 219, 220, 296]),
    "request_4": ("I need a defensive midfielder with a high willingness to run and good tackling skills", [1, 2, 5, 7, 25, 60, 63, 71, 72, 73, 110, 116, 142, 145, 183, 184, 185, 196, 234, 235, 245, 257, 269, 270, 272, 287]),
    "request_5": ("Goalkeeper with quick reflexes and good penalty area control", [31, 78])
}


#embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
embeddings = HuggingFaceEmbeddings(  
    model_name=model_name,  
    model_kwargs=model_kwargs,  
    encode_kwargs=encode_kwargs,  
)

# Connect to Milvus
connection_args = {'uri': VECTOR_STORE_URI}
vectorstore = Milvus(
    embedding_function=embeddings,
    connection_args=connection_args,
    collection_name=COLLECTION_NAME,
    vector_field="embeddings",
    primary_field="id",
    auto_id=True
)

retriever = vectorstore.as_retriever(search_kwargs={'k': 30})

# Load your JSON data
with open('retriever_test_reports.json', 'r') as file:
    data = json.load(file)

# Extract texts and player IDs
documents = [report['text'] for report in data]
player_ids = [report['report_number'] for report in data]

# Tokenize the documents for BM25
tokenized_corpus = [doc.split(" ") for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)

# Function to perform BM25 search
def bm25_search(query):
    tokenized_query = query.split(" ")
    doc_scores = bm25.get_scores(tokenized_query)
    # Get the indices of the top 30 scores
    top_n_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:30]
    return [player_ids[i] for i in top_n_indices]


# Function to query Milvus
def query_milvus(request_text):
    # Search for similar embeddings in Milvus
    results = retriever.invoke(request_text)
    
    # Extract player_ids from results
    player_ids = [result.metadata['id'] for result in results]
    return player_ids

# Calculate metrics
def calculate_metrics(request_text, validation_list, search_function):
    # Get the response (player list) from the search function
    response_ids = search_function(request_text)
    
    # Obtain the set of all possible player IDs 
    all_player_ids = range(1, 301)  # Comprehensive set of all player IDs in your database
    
    # Create binary lists for the true and predicted values
    y_true = [1 if player_id in validation_list else 0 for player_id in all_player_ids]
    y_pred = [1 if player_id in response_ids else 0 for player_id in all_player_ids]
    
    # Calculate Precision, Recall, and F1-Score
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred, zero_division=1)
    
    return precision, recall, f1

# Loop through requests and calculate metrics for both BM25 and Milvus
for req_name, (req_text, req_validation) in requests.items():
    bm25_precision, bm25_recall, bm25_f1 = calculate_metrics(req_text, req_validation, bm25_search)
    milvus_precision, milvus_recall, milvus_f1 = calculate_metrics(req_text, req_validation, query_milvus)
    
    print(f"{req_name} - BM25 - Precision: {bm25_precision:.2f}, Recall: {bm25_recall:.2f}, F1-Score: {bm25_f1:.2f}")
    print(f"{req_name} - Milvus - Precision: {milvus_precision:.2f}, Recall: {milvus_recall:.2f}, F1-Score: {milvus_f1:.2f}")

