import json
import random
from collections import defaultdict

# Load existing scouting reports
with open('scouting-llm/data/uhh_data_prod.json', 'r') as f:
    scouting_reports = json.load(f)

# Group the reports by player_id and player_transfermarkt_id
player_reports = defaultdict(list)
for report in scouting_reports:
    player_reports[(report['player_id'], report['player_transfermarkt_id'])].append(report)

# Choose random 100 players
selected_players = random.sample(list(player_reports.keys()), 100)

# Create the test dataset
test_dataset = []
for player_key in selected_players:
    test_dataset.extend(player_reports[player_key])

# Save the test dataset to a new JSON file
with open('test_scouting_reports.json', 'w') as f:
    json.dump(test_dataset, f, indent=2)

print("Test dataset with 100 players successfully created.")
