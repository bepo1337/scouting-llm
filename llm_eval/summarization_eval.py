import sys
import os

import bertscore
import rougescore
import player_count

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import model_definitions
import prompt_templates
import model_structure
from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from llm_eval.document_formats import format_documents_v01
import json
from dotenv import load_dotenv
import os
from langchain_openai import AzureChatOpenAI



load_dotenv()

MODELS_LOCAL = ["mistral", "llama3"]
# MODELS = ["mistral"]
MODELS_AZURE = ["gpt-4o", "gpt-4", "gpt-35-turbo"]
PROMPT_TEMPLATES = [prompt_templates.v005, prompt_templates.v006, prompt_templates.v007, prompt_templates.v008, prompt_templates.v009]
# PROMPT_TEMPLATES = [prompt_templates.v006]
# file_name = "test_structure.json"
file_name = "data/new_data_prod.json"
# file_name = "data/new_data_single_prod.json"
parser = JsonOutputParser(pydantic_object=model_definitions.ListPlayerResponse)
# TRUE = USES AZURE CLOUD
use_azure_model = True
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
OPENAI_API_VERSION = os.getenv('OPENAI_API_VERSION')


def get_reports_from_context_for_player(query_and_retrieved_doc_dict: model_structure.QueryAndRetrievedDocuments,
                                        player_id: str) -> str:
    query_and_retrieved_docs = model_structure.QueryAndRetrievedDocuments.parse_obj(query_and_retrieved_doc_dict)
    return_string = "\t"
    for doc in query_and_retrieved_docs.retrieved_documents:
        if doc.metadata['player_transfermarkt_id'] == player_id:
            return_string += "Report: " + doc.page_content + "\n\t"

    return return_string


def player_ids_in_retrieved_docs(query_and_retrieved_doc_dict: model_structure.QueryAndRetrievedDocuments) -> int:
    query_and_retrieved_docs = model_structure.QueryAndRetrievedDocuments.parse_obj(query_and_retrieved_doc_dict)

    player_id_map = {}
    for doc in query_and_retrieved_docs.retrieved_documents:
        player_id_map[doc.metadata['player_transfermarkt_id']] = ""

    return len(player_id_map)


list_of_test_inputs = model_structure.load_inputs(file_name)


def get_model_answer(current_model, current_prompt) -> model_definitions.ListPlayerResponse:
    model_answer = current_model.invoke(current_prompt)
    if use_azure_model:
        model_json_answer = json.loads(model_answer.content)
    else:
        model_json_answer = json.loads(model_answer)
    player_response = model_definitions.ListPlayerResponse(**model_json_answer)
    return player_response


def apply_metrics(predictions, references, players_in_list_from_references):
    bertscore.apply_bertscore(predictions, references)
    rougescore.apply_rouge_score(predictions, references)
    player_count.print_player_counts(players_in_list_from_references)


def setup_model(model_name: str):
    if use_azure_model:
        return AzureChatOpenAI(openai_api_key=AZURE_OPENAI_API_KEY, deployment_name=model_name,
                              model_kwargs={"response_format": {"type": "json_object"}})
    else:
        return Ollama(model=modelElement, format="json")


models = []
if use_azure_model:
    models = MODELS_AZURE
else:
    models = MODELS_LOCAL


# TODO get references list once and not in each model loop
for modelElement in models:
    with open('results.txt', 'a') as file:
        file.write(f"\n------\nCalculating metrics for model '{modelElement}'")
    print(f"\n------\nStarting calculating metrics for model '{modelElement}'...")
    model = setup_model(modelElement)


    for template in PROMPT_TEMPLATES:
        with open('results.txt', 'a') as file:
            file.write(f"\n------\nCalculating metrics for model '{modelElement}' with template '{template}'")
        print(f"\n------\nStarting calculating metrics for model '{modelElement}' with template '{template}'")
        prompt_template = template
        prompt_for_llm = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question", "format_instructions"],
        )

        predictions = []  # Model summaries
        references = []  # Our provided context data
        players_in_list_from_references = []  # Percentage of players the model had in its response wrt to the unique players in the context

        # kann man hier evtl auch die loop auslagern für code clarity?
        for single_instance_dict in list_of_test_inputs.data:
            single_input = model_structure.QueryAndRetrievedDocuments.parse_obj(single_instance_dict)
            print(f"start computing new input with query '{single_input.query}'...")
            formatted_context_string = format_documents_v01(single_input.retrieved_documents)

            # This is variable per prompt template and how we define the context (ie merging reports or not). Could also change the format instructions.
            prompt_injection = {"context": formatted_context_string, "question": single_input.query,
                                "format_instructions": parser.get_format_instructions()}
            prompt_for_llm = prompt_template.format(**prompt_injection)

            list_of_players_from_model = get_model_answer(model, prompt_for_llm)

            # craete 2 lists for the context and the corresponding llm-answer
            for player in list_of_players_from_model.list:
                model_summary = player.report_summary
                context_reports = get_reports_from_context_for_player(single_instance_dict, str(player.player_id))
                print(f"---\nModel answer and context for player_id: {player.player_id}:\n")
                print("model_summary: \n\t", model_summary)
                print("initial reports: \n", context_reports)

                predictions.append(model_summary)
                references.append(context_reports)

            # print comparison between players in context and lenght of player_response.list
            model_resp_player_count = len(list_of_players_from_model.list)
            unique_player_ids = player_ids_in_retrieved_docs(single_instance_dict)
            print(f"Expected {unique_player_ids} players in response, got {model_resp_player_count}")

            # TODO how to handle further?
            if model_resp_player_count > unique_player_ids:
                print(f"more entries in model response than in context: {list_of_players_from_model.list} for input {single_instance_dict}")
            players_in_list_from_references.append(model_resp_player_count / unique_player_ids)


        print(f"Calculating metrics...")
        apply_metrics(predictions, references, players_in_list_from_references)
        print(f"Finished calculating metrics for model '{modelElement}' with template '{template}\n\n")

    print(f"Finished calculating metrics for model '{modelElement}'")
