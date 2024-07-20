from chain import invoke_chain
from flask import jsonify
import json
from sklearn.metrics import precision_score, recall_score, f1_score

# Test requests and their respective validation lists of player_transfermarkt_ids
request_1 = "I am looking for an attacking midfielder with excellent ball control and strong passing skills, who is able to control the game and play creative passes."
request_1_validation = [981675, 865300, 807488, 670105, 690838, 589327, 1082852, 1240394, 392093, 555106, 368942, 400649, 989926, 1029814, 574658, 653687, 1077154, 658848, 450237, 207246, 651569, 840008, 626301, 223953, 891569, 63463]

request_2 = "I need a robust central defender with good heading and tackling ability who also has great anticipation and vision."
request_2_validation = [875118, 909249, 591848, 626678, 1031813, 661245, 562383, 1074505, 996926, 742857, 665541, 495726, 652362, 271323]

request_3 = "I am looking for a fast winger with high dribbling power and precise crosses who can play on both the right and left side."
request_3_validation = [451353, 309505, 659709, 345343, 368942, 583822, 844436, 988946, 945001, 989926, 1111047, 450237, 742857, 410456, 331739, 330699, 1165987, 657743, 856956]

request_4 = "I need a defensive midfielder with a high willingness to run and good tackling skills who can also play an important role in building up play."
request_4_validation = [981675, 807488, 670105, 909249, 690838, 509747, 1031813, 1045884, 400649, 664708, 658848, 665541, 891569, 271323]

request_5 = "I'm looking for a goalkeeper with quick reflexes and good penalty area control, who also has a strong player opening and precise shots."
request_5_validation = [378521]

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