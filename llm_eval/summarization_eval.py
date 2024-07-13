import sys
import os

import bertscore
import rougescore
import player_count

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

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


MODEL = "mistral"  # can be replaed by grid search later
prompt_template = prompt_templates.v005  # can be replaced by GS later
file_name = "test_structure.json"
#file_name = "test_prod.json"
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


def get_reports_from_context_for_player(query_and_retrieved_doc_dict: QueryAndRetrievedDocuments, player_id: str) -> str:
    query_and_retrieved_docs = QueryAndRetrievedDocuments.parse_obj(query_and_retrieved_doc_dict)
    return_string = "\t"
    for doc in query_and_retrieved_docs.retrieved_documents:
        if doc.metadata['player_transfermarkt_id'] == player_id:
            return_string += "Report: " + doc.page_content + "\n\t"

    return return_string

def get_average(float_list: List[float]) -> float:
    return sum(float_list) / len(float_list)


def player_ids_in_retrieved_docs(query_and_retrieved_doc_dict: QueryAndRetrievedDocuments) -> int:
    query_and_retrieved_docs = QueryAndRetrievedDocuments.parse_obj(query_and_retrieved_doc_dict)

    player_id_map = {}
    for doc in query_and_retrieved_docs.retrieved_documents:
        player_id_map[doc.metadata['player_transfermarkt_id']] = ""

    print(player_id_map)
    return len(player_id_map)


list_of_test_inputs = load_inputs()

predictions = []  # Model summaries
references = []  # Our provided context data
players_in_list_from_references = [] # Percentage of players the model had in its response wrt to the unique players in the context


def get_model_answer(current_model, current_prompt) -> model_definitions.ListPlayerResponse:
    model_answer = current_model.invoke(current_prompt)
    model_json_answer = json.loads(model_answer)
    player_response = model_definitions.ListPlayerResponse(**model_json_answer)
    return player_response


for singleInput in list_of_test_inputs:
    print("start computing new input...")
    # the following 2 lines only have to be done once in the beginning. Although doesnt matter if they run every time. But would be cleanerw
    actual_instance_of_input = QueryAndRetrievedDocuments.parse_obj(singleInput)
    formatted_context_string = format_documents_v01(actual_instance_of_input.retrieved_documents)

    # This is variable per prompt template and how we define the context (ie merging reports or not). Could also change the format instructions.
    prompt_injection = {"context": formatted_context_string, "question": actual_instance_of_input.query,
                        "format_instructions": parser.get_format_instructions()}
    prompt_for_llm = prompt_template.format(**prompt_injection)

    player_response = get_model_answer(model, prompt_for_llm)

    # craete 2 lists for the context and the corresponding llm-answer
    for player in player_response.list:
        model_summary = player.report_summary
        context_reports = get_reports_from_context_for_player(singleInput, str(player.player_id))
        print("---\nModel answer and context:\n")
        print("model_summary: \n\t", model_summary)
        print("initial reports: \n", context_reports)

        predictions.append(model_summary)
        references.append(context_reports)

    # print comparison between players in context and lenght of player_response.list
    model_resp_player_count = len(player_response.list)
    unique_player_ids = player_ids_in_retrieved_docs(singleInput)
    players_in_list_from_references.append(model_resp_player_count/unique_player_ids)



    # for every player now check the metrics
    # for list_item in model_json_answer:

bertscore.apply_bertscore(predictions, references)
rougescore.apply_rouge_score(predictions, references)
player_count.print_player_counts(players_in_list_from_references)
