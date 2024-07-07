import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import model_definitions
import prompt_templates
from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from document_formats import format_documents_v01
from pydantic.v1 import BaseModel
from langchain_core.documents import Document
from typing import List
import json
from evaluate import load

MODEL = "mistral"  # can be replaed by grid search later
prompt_template = prompt_templates.v005  # can be replaced by GS later
file_name = "test_structure.json"
parser = JsonOutputParser(pydantic_object=model_definitions.ListPlayerResponse)
prompt_for_llm = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question", "format_instructions"],
)

model = Ollama(model=MODEL, format="json")


# Define model
class QueryAndRetrievedDocuments(BaseModel):
    query: str
    retrieved_documents: List[Document]


class ListOfTestInputs(BaseModel):
    data: List[QueryAndRetrievedDocuments]


# Load the the queries and corresponding retrived documents
def load_inputs() -> ListOfTestInputs:
    with open(file_name, "r") as file:
        json_data = file.read()
        parsed_data = json.loads(json_data)
        return parsed_data


def get_reports_from_context(query_and_retrieved_doc_dict: QueryAndRetrievedDocuments, player_id: str) -> str:
    query_and_retrieved_docs = QueryAndRetrievedDocuments.parse_obj(query_and_retrieved_doc_dict)
    return_string = "\t"
    for doc in query_and_retrieved_docs.retrieved_documents:
        if doc.metadata['player_transfermarkt_id'] == player_id:
            return_string += "Report: " + doc.page_content + "\n\t"

    return return_string


list_of_test_inputs = load_inputs()

for singleInput in list_of_test_inputs:
    actual_instance_of_input = QueryAndRetrievedDocuments.parse_obj(singleInput)
    formatted_context_string = format_documents_v01(actual_instance_of_input.retrieved_documents)

    prompt_injection = {"context": formatted_context_string, "question": actual_instance_of_input.query,
                        "format_instructions": parser.get_format_instructions()}
    prompt_for_llm = prompt_template.format(**prompt_injection)
    print(prompt_for_llm)

    model_answer = model.invoke(prompt_for_llm)
    # print(model_answer)

    model_json_answer = json.loads(model_answer)
    # print(model_json_answer)

    player_response = model_definitions.ListPlayerResponse(**model_json_answer)

    # Metrics
    bertscore_metrics = []
    berscore = load("bertscore")
    for player in player_response.list:
        model_summary = player.report_summary
        context_reports = get_reports_from_context(singleInput, str(player.player_id))
        print("comparison now:")
        print("model_summary: \n\t", model_summary)
        print("initial reports: \n", context_reports)

        # bertscore
        predictions = [model_summary]
        references = [context_reports]
        # other model such as "roberta-large" is better, but larger obv (distilbert... takes 268MB vs roberta-large is 1.4GB)
        print("bert score: ",
              berscore.compute(predictions=predictions, references=references, model_type="distilbert-base-uncased"))
    # for every player now check the metrics
    # for list_item in model_json_answer:
