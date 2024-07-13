from pydantic import BaseModel, Field
from typing import List

class PlayerResponse(BaseModel):
    player_id: int = Field(description="ID of the player")
    report_summary: str = Field(name="report_summary",
                                description="Summary of the reports that have the same player id")


# We want to get a list of players
class ListPlayerResponse(BaseModel):
    list: List[PlayerResponse]
