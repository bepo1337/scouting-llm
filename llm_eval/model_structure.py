from pydantic.v1 import BaseModel
from typing import List
from langchain_core.documents import Document
import json

# Define model
class QueryAndRetrievedDocuments(BaseModel):
    query: str
    retrieved_documents: List[Document]


class ListOfTestInputs(BaseModel):
    data: List[QueryAndRetrievedDocuments]


class GoldenSummaryAndRetrievedDocuments(BaseModel):
    golden_summary: str
    retrieved_documents: List[Document]


class ListOfGoldenSummaryAndRetrievedDocuments(BaseModel):
    data: List[GoldenSummaryAndRetrievedDocuments]



# Load the the queries and corresponding retrieved documents
def load_inputs(file_name) -> ListOfTestInputs:
    with open(file_name, "r") as file:
        json_data = file.read()
        parsed_data = json.loads(json_data)
        list_of_inputs = ListOfTestInputs(**parsed_data)
        return list_of_inputs

def load_or_create(file_name) -> ListOfTestInputs:
    try:
        parsed_date = load_inputs(file_name)
        return parsed_date
    except FileNotFoundError:
        print(f"{file_name} failed to load, it will overwrite on saving or create the file when typing 'done'. "
              f"Exit with CTRl+C otherwise.")
        return ListOfTestInputs(data=[])