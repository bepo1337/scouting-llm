from pymilvus import connections, Collection
from sklearn.metrics import precision_score, recall_score
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

# embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
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

retriever = vectorstore.as_retriever(search_kwargs={'k': 40})

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
    # Get the indices of the top 40 scores
    top_n_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:40]
    return [player_ids[i] for i in top_n_indices]

# Function to query Milvus
def query_milvus(request_text):
    # Search for similar embeddings in Milvus
    results = retriever.invoke(request_text)
    
    # Extract player_ids from results
    player_ids = [result.metadata['id'] for result in results]
    return player_ids

# Calculate precision and recall at various k
def calculate_metrics_at_k(request_text, validation_list, search_function, ks):
    # Get the response (player list) from the search function
    response_ids = search_function(request_text)
    
    metrics = {}
    
    for k in ks:
        # Consider only the top-k results
        top_k_response = response_ids[:k]
        
        # Calculate Precision@k and Recall@k
        precision_at_k = len(set(top_k_response) & set(validation_list)) / k
        recall_at_k = len(set(top_k_response) & set(validation_list)) / len(validation_list)
        
        metrics[k] = {
            'precision': precision_at_k,
            'recall': recall_at_k
        }
    
    return metrics

# Define the values of k you want to test
ks = [1, 5, 10, 20, 30, 40]

# Initialize accumulators for average precision and recall
bm25_precision_sums = {k: 0 for k in ks}
bm25_recall_sums = {k: 0 for k in ks}
milvus_precision_sums = {k: 0 for k in ks}
milvus_recall_sums = {k: 0 for k in ks}
num_requests = len(requests)

# Loop through requests and calculate metrics for both BM25 and Milvus
for req_name, (req_text, req_validation) in requests.items():
    bm25_metrics = calculate_metrics_at_k(req_text, req_validation, bm25_search, ks)
    milvus_metrics = calculate_metrics_at_k(req_text, req_validation, query_milvus, ks)
    
    # Accumulate the metrics for average calculation
    for k in ks:
        bm25_precision_sums[k] += bm25_metrics[k]['precision']
        bm25_recall_sums[k] += bm25_metrics[k]['recall']
        milvus_precision_sums[k] += milvus_metrics[k]['precision']
        milvus_recall_sums[k] += milvus_metrics[k]['recall']
    
    # Print individual request metrics (optional)
    print(f"{req_name} - BM25 Metrics:")
    for k in ks:
        print(f"  @k={k} - Precision: {bm25_metrics[k]['precision']:.2f}, Recall: {bm25_metrics[k]['recall']:.2f}")
    
    print(f"{req_name} - Milvus Metrics:")
    for k in ks:
        print(f"  @k={k} - Precision: {milvus_metrics[k]['precision']:.2f}, Recall: {milvus_metrics[k]['recall']:.2f}")

# Calculate and print the average precision and recall across all requests
print("\nAverage BM25 Metrics:")
for k in ks:
    avg_precision = bm25_precision_sums[k] / num_requests
    avg_recall = bm25_recall_sums[k] / num_requests
    print(f"  @k={k} - Average Precision: {avg_precision:.2f}, Average Recall: {avg_recall:.2f}")

print("\nAverage Milvus Metrics:")
for k in ks:
    avg_precision = milvus_precision_sums[k] / num_requests
    avg_recall = milvus_recall_sums[k] / num_requests
    print(f"  @k={k} - Average Precision: {avg_precision:.2f}, Average Recall: {avg_recall:.2f}")
