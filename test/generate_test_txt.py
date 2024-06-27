import json
from collections import defaultdict

# Load the test dataset
with open('test_scouting_reports.json', 'r') as f:
    test_scouting_reports = json.load(f)

# Group the reports by player
player_reports = defaultdict(list)
for report in test_scouting_reports:
    player_key = (report['player_id'], report['player_transfermarkt_id'])
    player_reports[player_key].append(report)

# Create the text file
with open('scouting_reports.txt', 'w') as f:
    for player_key, reports in player_reports.items():
        player_id, player_transfermarkt_id = player_key
        f.write(f"Player ID: {player_id}\n")
        f.write(f"Player Transfermarkt ID: {player_transfermarkt_id}\n\n")
        for report in reports:
            main_position = report['main_position']
            played_position = report['played_position']
            f.write(f"Main Position: {main_position}\n")
            f.write(f"Played Position: {played_position}\n")
            f.write(f"{report['text']}\n\n")
        f.write("R1 [ ], R2 [ ], R3 [ ], R4 [ ], R5 [ ]\n\n")
        f.write("\n")

print("Text file with scouting reports for annotation successfully created.")
