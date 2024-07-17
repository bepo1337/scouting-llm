from pydantic.v1 import BaseModel
from typing import List
from langchain_core.documents import Document
import json

# Define model
class QueryAndRetrievedDocuments(BaseModel):
    query: str
    retrieved_documents: List[Document]
    # TODO add golden summary dict: player_id --> golden summary


class ListOfTestInputs(BaseModel):
    data: List[QueryAndRetrievedDocuments]


# Load the the queries and corresponding retrieved documents
def load_inputs(file_name) -> ListOfTestInputs:
    with open(file_name, "r") as file:
        json_data = file.read()
        parsed_data = json.loads(json_data)
        return parsed_data