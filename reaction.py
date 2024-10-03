import json
import os
from typing import List

from pydantic import BaseModel

logFile = "data/reaction_log_prod.json"

class Reaction(BaseModel):
    query: str
    playerID: int
    reaction: str
    summary: str


# We want to get a list of players
class Reactions(BaseModel):
    list: List[Reaction]


def create_log_file_if_not_exist():
    """Creates the log file if it doesnt yet exist"""
    if os.path.isfile(logFile):
        print(f"The file '{logFile}' exists.")
    else:
        with open(logFile, 'w') as file:
            reactions = Reactions(list=[])
            file.write(reactions.model_dump_json())
        print(f"The file '{logFile}' was created.")



def append_to_log(request_json):
    """Parses the JSON and appends the reaction to the log file"""
    with(open(logFile, 'r')) as file:
        reactions_json = json.load(file)

    reactions = Reactions.parse_obj(reactions_json)
    reaction_entry = Reaction.parse_obj(request_json)
    reactions.list.append(reaction_entry)

    with open(logFile, "w") as file:
        file.write(reactions.model_dump_json())


def log_reaction(request_json):
    """Logs the reaction to the log file"""
    create_log_file_if_not_exist()
    append_to_log(request_json)