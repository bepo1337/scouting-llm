import json
import os
from typing import List

from pydantic import BaseModel

# Path to the log file where reactions will be saved
logFile = "data/reaction_log_prod.json"

# Structure of the Reaction entry
class Reaction(BaseModel):
    query: str
    playerID: int
    reaction: str
    summary: str


# Class to hold a list of Reaction objects
class Reactions(BaseModel):
    list: List[Reaction]

# Function to create the log file if it doesn't exist
def create_log_file_if_not_exist():
    if os.path.isfile(logFile):
        print(f"The file '{logFile}' exists.")
    else:
        with open(logFile, 'w') as file:
            reactions = Reactions(list=[])
            file.write(reactions.model_dump_json())
        print(f"The file '{logFile}' was created.")


# Function to parse a JSON and append a reaction to the log file
def append_to_log(request_json):
    with(open(logFile, 'r')) as file:
        reactions_json = json.load(file)

    reactions = Reactions.parse_obj(reactions_json)
    reaction_entry = Reaction.parse_obj(request_json)
    reactions.list.append(reaction_entry)

    with open(logFile, "w") as file:
        file.write(reactions.model_dump_json())

# Log the reaction to the log file if the log file exists. Otherwise, create the log file and log the reaction.
def log_reaction(request_json):
    create_log_file_if_not_exist()
    append_to_log(request_json)