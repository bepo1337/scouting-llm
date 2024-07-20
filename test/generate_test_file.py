import json

# Load JSON data from a file
with open('min-project-4-at-2024-07-16-22-48-47bf8c02.json', 'r') as file:
    data = json.load(file)

# Predefined requests and report number mappings
requests = {
    "I am looking for an attacking midfielder with excellent ball control and strong passing skills": [],
    "Robust central defender with good heading and tackling ability": [],
    "I am looking for a fast winger with high dribbling power and precise crosses who can play on both the right and left side.": [],
    "I need a defensive midfielder with a high willingness to run and good tackling skills": [],
    "Goalkeeper with quick reflexes and good penalty area control": []
}

# Process the data
for item in data:
    if "selection" in item:
        if isinstance(item["selection"], dict):
            # Multiple selections
            for choice in item["selection"]["choices"]:
                if choice in requests:
                    requests[choice].append(item["report_number"])
        else:
            # Single selection
            if item["selection"] in requests:
                requests[item["selection"]].append(item["report_number"])

# Print results
for i, (query, report_numbers) in enumerate(requests.items(), start=1):
    print(f"request_{i} = \"{query}\"")
    print(f"request_{i}_validation = {report_numbers}\n")