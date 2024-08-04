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
    GOALKEEPER = 'Goalkeeper',
    CENTER_BACK = 'Centre back',
    LEFT_CENTER_BACK = 'Left center back',
    RIGHT_CENTER_BACK = 'Right center back',
    LEFT_BACK = 'Left back',
    LEFT_WING_BACK = 'Left wing back',
    RIGHT_BACK = 'Right back',
    RIGHT_WING_BACK = 'Right wing back',
    DEFENSIVE_MIDFIELD = 'Defensive midfield',
    DEFENSIVE_MIDFIELD_LEFT = 'Defensive midfield left',
    DEFENSIVE_MIDFIELD_RIGHT = 'Defensive midfield right',
    CENTRAL_MIDFIELD = 'Central midfield',
    CENTRAL_MIDFIELD_RIGHT = 'Central midfield right',
    CENTRAL_MIDFIELD_LEFT = 'Central midfield left',
    RIGHT_MIDFIELD = 'Right midfield',
    RIGHT_MIDFIELD_OFFENSIVE = 'Right midfield offensive',
    LEFT_MIDFIELD = 'Left midfield',
    LEFT_MIDFIELD_OFFENSIVE = 'Left midfield offensive',
    ATTACKING_MIDFIELD = 'Attacking midfield',
    ATTACKING_MIDFIELD_LEFT = 'Attacking midfield left',
    ATTACKING_MIDFIELD_RIGHT = 'Attacking midfield right',
    HANGING_TOP = 'Hanging top',
    LEFT_WING = 'Left wing',
    RIGHT_WING = 'Right wing',
    CENTER_FORWARD = 'Center forward',
    RIGHT_FORWARD = 'Right forward',
    LEFT_FORWARD = 'Left forward',
    SUBSTITUTE = 'Substitute'
