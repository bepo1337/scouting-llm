import model_structure
from langchain_community.llms.ollama import Ollama
import json
from dotenv import load_dotenv
import os
from langchain_openai import AzureChatOpenAI
import bertscore
import rougescore
import bleu

load_dotenv()
MODELS_LOCAL = ["mistral", "llama3"]
MODELS_AZURE = ["gpt-4o", "gpt-4", "gpt-35-turbo"]
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
OPENAI_API_VERSION = os.getenv('OPENAI_API_VERSION')
test_data_file_name = "data/summarization_with_golden_summaries_prod.json"
use_azure_model = True
results_filename = "summary_results.txt"

def setup_model(model_name: str):
    if use_azure_model:
        return AzureChatOpenAI(openai_api_key=AZURE_OPENAI_API_KEY, deployment_name=model_name)
    else:
        return Ollama(model=model_name)

def load_test_data():
    with open(test_data_file_name, 'r') as file:
        summariesWithReportsJSON = json.load(file)
        summariesWithReports = model_structure.ListOfGoldenSummaryAndRetrievedDocuments(**summariesWithReportsJSON)
        return summariesWithReports



def get_golden_summaries(listOfGoldenSummariesAndDocs: model_structure.ListOfGoldenSummaryAndRetrievedDocuments):
    summaries = []
    for val in listOfGoldenSummariesAndDocs.data:
        summaries.append(val.golden_summary)

    return summaries


def get_model_answer(current_model, current_prompt) -> str:
    model_answer = current_model.invoke(current_prompt)
    if use_azure_model:
        return model_answer.content
    else:
        return model_answer


def apply_metrics(predictions, references):
    bertscore.apply_bertscore(predictions, references, results_filename)
    rougescore.apply_rouge_score(predictions, references, results_filename)
    bleu.apply_bleu_score(predictions, references, results_filename)


test_data = load_test_data()
golden_summaries = get_golden_summaries(test_data)
# For all models
models = []
if use_azure_model:
    models = MODELS_AZURE
else:
    models = MODELS_LOCAL

for modelElement in models:
    with open(results_filename, 'a') as file:
        file.write(f"\n------\nCalculating metrics for model '{modelElement}'")
    print(f"\n------\nStarting calculating metrics for model '{modelElement}'...")
    model = setup_model(modelElement)
    model_answers = []
    for element in test_data.data:
        prompt = "Summarize the following soccer reports about a player. Return only the summary and nothing else.\n\n"
        reportCount = 1
        for report in element.retrieved_documents:
            prompt += f"Report {reportCount}: {report.page_content}\n\n"
            reportCount += 1

        answer = get_model_answer(model, prompt)
        model_answers.append(answer)

    apply_metrics(predictions=model_answers, references=golden_summaries)