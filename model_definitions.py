from enum import Enum

from pydantic import BaseModel, Field
from typing import List


class PlayerResponse(BaseModel):
    player_id: int = Field(description="ID of the player")
    report_summary: str = Field(name="report_summary",
                                description="Summary of the reports that have the same player id")


# We want to get a list of players
class ListPlayerResponse(BaseModel):
    list: List[PlayerResponse]


class Positions(Enum):
    goalkeeper = 'Goalkeeper',
    centerback = 'Centre back',
    leftcenterback = 'Left center back',
    rightcenterback = 'Right center back',
    leftback = 'Left back',
    leftwingback = 'Left wing back',
    rightback = 'Right back',
    rightwingback = 'Right wing back',
    defensivemidfield = 'Defensive midfield',
    centralmidfield = 'Central midfield',
    rightmidfield = 'Right midfield',
    leftmidfield = 'Left midfield',
    attackingmidfield = 'Attacking midfield',
    hangingtop = 'Hanging top',
    leftwing = 'Left wing',
    rightwing = 'Right wing',
    centerforward = 'Center forward',
    substitute = 'Substitute'

def get_position_key_from_value(value):
    for position in Positions:
        # otherwise its a tuple somehow? works
        if position.value[0] == value:
            return position.name
    return None
