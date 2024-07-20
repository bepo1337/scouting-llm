from chain import invoke_chain
from flask import jsonify
import json
from sklearn.metrics import precision_score, recall_score, f1_score

request_1 = "I am looking for an attacking midfielder with excellent ball control and strong passing skills"
request_1_validation = [9, 35, 36, 37, 39, 49, 52, 59, 60, 61, 77, 81, 82, 85, 86, 89, 98, 99, 103, 116, 140, 153, 154, 158, 159, 171, 174, 178, 193, 194, 232, 245, 249, 250, 256, 259, 279]

request_2 = "Robust central defender with good heading and tackling ability"
request_2_validation = [13, 18, 21, 40, 42, 46, 48, 57, 63, 64, 73, 74, 76, 95, 96, 111, 113, 115, 121, 122, 173, 196, 198, 221, 244, 275, 276, 292, 293, 294]

request_3 = "I am looking for a fast winger with high dribbling power and precise crosses who can play on both the right and left side."
request_3_validation = [3, 5, 12, 65, 66, 81, 82, 87, 89, 92, 112, 124, 132, 133, 134, 135, 136, 137, 139, 140, 170, 180, 181, 203, 205, 211, 212, 216, 217, 218, 219, 220, 296]

request_4 = "I need a defensive midfielder with a high willingness to run and good tackling skills"
request_4_validation = [1, 2, 5, 7, 25, 60, 63, 71, 72, 73, 110, 116, 142, 145, 183, 184, 185, 196, 234, 235, 245, 257, 269, 270, 272, 287]

request_5 = "Goalkeeper with quick reflexes and good penalty area control"
request_5_validation = [31, 78]

def calculate_metrics(request, validation_list):
    # Get the response (player list) by invoke_chain
    response = invoke_chain(request)
    print ("Response: ", response)
    
    # Get the relevant player ids from the response 
    response_ids = [player['player_id'] for player in response['list']]
    
    # Create binary lists for the true and predicted values
    y_true = [1 if player_id in validation_list else 0 for player_id in validation_list]
    y_pred = [1 if player_id in response_ids else 0 for player_id in validation_list]
    
    # Calculate Precision, Recall und F1-Score
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    return precision, recall, f1

# Get the validation metrics for a specific request and validation set
precision, recall, f1 = calculate_metrics(request_1, request_1_validation)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")