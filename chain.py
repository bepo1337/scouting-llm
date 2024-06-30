from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
import json
from collections import defaultdict

import model_definitions
import prompt_templates
from langchain_community.vectorstores import Milvus
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser

MODEL = "mistral"
EMBEDDING_MODEL = "nomic-embed-text"
DIMENSIONS = 768
COLLECTION_NAME = "test_scouting"
VECTOR_STORE_URI = "http://localhost:19530"
OLLAMA_URI = "http://localhost:11434/v1"
COUNT_RETRIEVED_DOCUMENTS = 20

#LLM https://api.python.langchain.com/en/latest/llms/langchain_community.llms.ollama.Ollama.html
model = Ollama(model=MODEL, format="json")
#Embedding
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

#Prompt
#https://python.langchain.com/v0.1/docs/modules/model_io/output_parsers/types/json/
parser = JsonOutputParser(pydantic_object=model_definitions.ListPlayerResponse)
prompt = PromptTemplate(
    template=prompt_templates.v003,
    input_variables=["context", "question"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

connection_args = {'uri': VECTOR_STORE_URI}
vectorstore = Milvus(
    embedding_function=embeddings,
    connection_args=connection_args,
    collection_name=COLLECTION_NAME,
    vector_field="embeddings",
    primary_field="id",
    auto_id=True
)

retriever = vectorstore.as_retriever(search_kwargs={'k': COUNT_RETRIEVED_DOCUMENTS})

##### Dont have to edit anything below this to change models
def format_docs(docs):
    # Create a dictionary to hold reports for each player ID
    player_reports = defaultdict(list)

    # Aggregate reports by player ID
    for doc in docs:
        player_id = doc.metadata['player_transfermarkt_id']
        report_content = doc.page_content
        player_reports[player_id].append(report_content)

    # Format the aggregated reports
    formatted_reports = []
    for player_id, reports in player_reports.items():
        formatted_report = f"Player ID: {player_id}\n"
        for i, report in enumerate(reports, 1):
            formatted_report += f"Report {i}: {report}\n"
        formatted_reports.append(formatted_report.strip())

    # Join all formatted reports into a single string
    return_string = "\n\n".join(formatted_reports)
    print(return_string)
    return return_string

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt        # Put retrieved context and passed through question into prompt template --> formatted prompt to llm
    | model         # Outputs LLM response
    | json.loads    # Parses response to json object
)

def invoke_chain(query: str) -> str:
    return rag_chain.invoke(query)

