from langchain_core.prompts import PromptTemplate



TUTORIAL_PROMPT = """
Human: You are an AI assistant, and provides answers to questions by using fact based and statistical information when possible.
Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
If you don't know the answer from the context, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

<question>
{question}
</question>

The response should be specific and use statistics or numbers when possible.

Assistant:"""

tutorial_prompt = PromptTemplate(
    template=TUTORIAL_PROMPT, input_variables=["context", "question"]
)