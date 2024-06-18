from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate

import prompt_templates
from langchain_community.vectorstores import Milvus
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


MODEL = "mistral"
EMBEDDING_MODEL = "nomic-embed-text"
DIMENSIONS = 768
COLLECTION_NAME = "scouting"
VECTOR_STORE_URI = "http://localhost:19530"
OLLAMA_URI = "http://localhost:11434/v1"

#LLM
model = Ollama(model=MODEL)

#Embedding
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

#Prompt
prompt = PromptTemplate(
    template=prompt_templates.STRUCTURE_AND_RANK, input_variables=["context", "question"]
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

retriever = vectorstore.as_retriever(search_kwargs={'k': 20})

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
        combined_reports = "\n".join(reports)
        formatted_reports.append(f"Player ID: {player_id}, Reports: {combined_reports}")
    
    # Join all formatted reports into a single string
    return_string = "\n\n".join(formatted_reports)
    print(return_string)
    return return_string

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

def invoke_chain(query: str) -> str:
    return rag_chain.invoke(query)


