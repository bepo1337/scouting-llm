from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
import json

import model_definitions
import prompt_templates
from langchain_community.vectorstores import Milvus
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser

MODEL = "mistral"
EMBEDDING_MODEL = "nomic-embed-text"
DIMENSIONS = 768
COLLECTION_NAME = "scouting"
VECTOR_STORE_URI = "http://localhost:19530"
OLLAMA_URI = "http://localhost:11434/v1"

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

retriever = vectorstore.as_retriever(search_kwargs={'k': 2})

##### Dont have to edit anything below this to change models
def format_docs(docs):
    returnString = "\n\n".join(f"Player ID: {doc.metadata['player_transfermarkt_id']}, Report-Content: " + doc.page_content for doc in docs)
    print(returnString)
    return returnString

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt        # Put retrieved context and passed through question into prompt template --> formatted prompt to llm
    | model         # Outputs LLM response
    | json.loads    # Parses response to json object
)

def invoke_chain(query: str) -> str:
    return rag_chain.invoke(query)

