# Running this file and entering queries will add test data to the file specified below.
# Type "done" when no more data is to be added
# New data will be saved after "done" was typed

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Milvus
import model_structure


COUNT_RETRIEVED_DOCUMENTS = 5
EMBEDDING_MODEL = "nomic-embed-text"
DIMENSIONS = 768
COLLECTION_NAME = "scouting"
VECTOR_STORE_URI = "http://localhost:19530"
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
connection_args = {'uri': VECTOR_STORE_URI}
file_name = "new_data_prod.json"

vectorstore = Milvus(
    embedding_function=embeddings,
    connection_args=connection_args,
    collection_name=COLLECTION_NAME,
    vector_field="embeddings",
    primary_field="id",
    auto_id=True
)

retriever = vectorstore.as_retriever(search_kwargs={'k': COUNT_RETRIEVED_DOCUMENTS})

test_data_file = model_structure.load_inputs(file_name)
test_data = test_data_file['data']

while True:
    # Prompt the user for input
    user_input = input("Enter query (type 'done' to finish): ")

    # Check if the user wants to stop the loop
    if user_input.lower() == 'done':
        break

    retrieved_docs = retriever.invoke(user_input)
    new_obj = model_structure.QueryAndRetrievedDocuments(query=user_input, retrieved_documents=retrieved_docs)
    test_data.append(new_obj)


new_list_of_test_pairs = model_structure.ListOfTestInputs(data=test_data)

with open(file_name, 'w') as file:
    print(f"writing to {file_name}...")
    file.write(new_list_of_test_pairs.json())
    print(f"successfully writen to {file_name}")
