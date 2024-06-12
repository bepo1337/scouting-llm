from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
import prompt_templates
from langchain_community.vectorstores import Milvus
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


MODEL = "mistral"
EMBEDDING_MODEL = "nomic-embed-text"
DIMENSIONS = 768
COLLECTION_NAME = "scouting"
VECTOR_STORE_URI = "http://localhost:19530"

#LLM
model = Ollama(model=MODEL)

#Embedding
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

#Prompt
prompt = prompt_templates.tutorial_prompt

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

##### Dont have to edit anything below this
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

def invoke_chain(query: str) -> str:
    return rag_chain.invoke(query)


