from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate

import prompt_templates
from langchain_community.vectorstores import Milvus
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import instructor
from pydantic import BaseModel, Field
from typing import List
from openai import OpenAI
import json


MODEL = "mistral"
EMBEDDING_MODEL = "nomic-embed-text"
COLLECTION_NAME = "scouting"
VECTOR_STORE_URI = "http://localhost:19530"
OLLAMA_URI = "http://localhost:11434/v1"

#Embedding
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)


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

# Define how the answer per player should look like
class PlayerResponse(BaseModel):
    player_id: int = Field(description="ID of the player")
    report_summary: str = Field(name="report_summary",
                                description="Summary of the reports that have the same player id")


# We want to get a list of players
class ListPlayerResponse(BaseModel):
    list: List[PlayerResponse]


instructor_client = instructor.from_openai(
    OpenAI(
        # default port of ollama
        base_url="http://localhost:11434/v1",
        # need to specify this, otherwise we get an error from the library
        api_key="ollama"
    ),
    mode=instructor.Mode.JSON
)


def format_docs(docs):
    joinedDocumentsAsString = "\n\n".join(f"Player ID: {doc.metadata['player_transfermarkt_id']}, Report-Content: " + doc.page_content for doc in docs)
    print(joinedDocumentsAsString)
    return joinedDocumentsAsString

def build_prompt(context: str, question: str) -> str:
        return f"""You are an assistant in football (soccer) scouting, and provides answers to questions by using fact based information.
    Use the following information to provide a concise answer to the question enclosed in <question> tags.
    If you don't know the answer from the context, just say that you don't know
    
    <context>
    {context}
    </context>
    
    <question>
    {question}
    </question>
    """


instructor_client = instructor.from_openai(
    OpenAI(
        # default port of ollama
        base_url="http://localhost:11434/v1",
        # need to specify this, otherwise we get an error from the library
        api_key="ollama"
    ),
    mode=instructor.Mode.JSON
)



def invoke_chain(query: str) -> str:
    documents = retriever.invoke(query)
    formatted_documents = format_docs(documents)

    final_prompt = build_prompt(context=formatted_documents, question=query)
    print(final_prompt)

    resp = instructor_client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": final_prompt,
        }],
        response_model=ListPlayerResponse,
    )
    model_json_str = resp.model_dump_json(indent=2)
    json_str = json.loads(model_json_str)

    return json_str


