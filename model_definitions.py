from enum import Enum

from pydantic import BaseModel, Field
from typing import List


class PlayerIDWithSummary(BaseModel):
    player_id: int = Field(description="ID of the player")
    report_summary: str = Field(name="report_summary",
                                description="Summary of the reports that have the same player id")


class PlayerIDWithSummaryAndFineGrainedReports(BaseModel):
    player_id: int = Field(description="ID of the player")
    report_summary: str = Field(name="report_summary",
                                description="Summary of the reports that have the same player id")
    fine_grained_reports: List[str] = Field(name="fine_grained_reports",
                                        description="Reports that were found in the fine grained search where we only look at single reports, not at summaries of players")


# We want to get a list of players
class ListPlayerResponse(BaseModel):
    list: List[PlayerIDWithSummaryAndFineGrainedReports]


class ComparePlayerRequestPayload(BaseModel):
    player_left: int
    player_right: int
    all: bool
    offensive: bool
    defensive: bool
    strenghts: bool
    weaknesses: bool
    other: bool
    otherText: str


class ComparePlayerResponsePayload(BaseModel):
    player_left: int
    player_left_name: str
    player_right: int
    player_right_name: str
    comparison: str

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
