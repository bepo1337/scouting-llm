from langchain_core.prompts import PromptTemplate

PROMPT_SUMMARY_INTO_STRUCTURE = """
Context is soccer and you are a assistant in scouting. I want you to summarize the following reports into a structured summary.
The structure for the summary is as follows:
- General text about the player
- Strengths
- Weaknesses
- Physical capabilities
- Offensive capabilities
- Defensive capabilities
- Other attributions

This is an example for a summary, but yours can vary in length and content depending on the reports in the context:
"He is widely regarded as one of the best defenders of the country. He still has a lot of potential for improvement though.

Strenghts: Very good in duels and opening passes
Weaknesses: Often times out of position, opening up a lot of space
Physical capabilities: Medium pace but a lot of endurance for intense games
Offensive capabilities: -
Defensive capabilities: Brings order to his teammates and wins a lot of duels. Very good at defending in aerial duels. 
Other attributions: Motivates his teammates and doesnt give up even when down by some goals
"
The reports about the player are the following:
<reports>
{reports}
</reports>
Do not make anything up that you dont see from the reports.
"""

PROMPT_SUMMARY_INTO_STRUCTURE_WITHOUT_EXAMPLE = """
Context is soccer and you are a assistant in scouting. I want you to summarize the following reports into a structured summary.
The structure for the summary is as follows:
- General text about the player
- Strengths
- Weaknesses
- Physical capabilities
- Offensive capabilities
- Defensive capabilities
- Other attributions

The reports about the player are the following:
<reports>
{reports}
</reports>
Do not make anything up that you dont see from the reports.
"""

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


STRUCTURE_AND_RANK = """
Human: You are an AI assistant in football (soccer) scouting, and provides answers to questions by using fact based and statistical information when possible.
Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
If you don't know the answer from the context, just say that you don't know, don't try to make up an answer.

<context>
{context}
</context>

<question>
{question}
</question>

The response should be specific and use statistics or numbers when possible.
The structure the response should be that you rank the players based on their reports and provide a short summary
of the reports from the context as follows:

1: Player <Player-ID-1>: <Summary-1>
2. Player <Player-ID-2>: <Summary-2>

Assistant:"""

# von https://www.youtube.com/watch?v=UVn2NroKQCw  ganz grob
youtube_template_string = """
You are an assistant in football (soccer) scouting, and provides answers to questions by using fact based information.
Use the following information to provide a concise answer to the question enclosed in <question> tags.
If you don't know the answer from the context, just say that you don't know

<context>
{context}
</context>

<question>
{question}
</question>

{format_instruction}
"""

v001 = """You are an assistant in football (soccer) scouting, and provides answers to questions by using fact based information.
    Use the following information to provide a concise answer to the question enclosed in <question> tags.
    If you don't know the answer from the context, just say that you don't know.
    
    <context>
    {context}
    </context>
    
    <question>
    {question}
    </question>
"""

v002 = """You are an assistant in football (soccer) scouting, and provides answers to questions by using fact based information.
    Use the following information to provide a concise answer to the question enclosed in <question> tags.
    If you don't know the answer from the context, just say that you don't know.

    <context>
    {context}
    </context>

    <question>
    {question}
    </question>
    
    The format of your answer should be a JSON list that for each elements has two entries:
    "player_id": integer
    "report_summary": string
    
    So create an entry in this json list for every unique player_id that you have in the <context> tags. Also create a list if you only have a single element.
"""

v003 = """You are an assistant in football (soccer) scouting, and provide answers to questions by using fact based information.
    Use the following information to provide a concise answer to the question enclosed in <question> tags.
    If you don't know the answer from the context, just say that you don't know.

    <context>
    {context}
    </context>

    <question>
    {question}
    </question>

   {format_instructions}
"""

v004 = """You are an assistant in football (soccer) scouting.
    Use the following information to provide a concise answer to the question enclosed in <question> tags.
    For your answer, paraphrase each report_summary at least a little bit and dont return it in its original form.

    <context>
    {context}
    </context>

    <question>
    {question}
    </question>

   {format_instructions}
"""

v005 = """You are an assistant in football (soccer) scouting.
    Use the following information to provide a concise answer to the question enclosed in <question> tags.
    Dont make up anything that you dont see from the context.
    
    <context>
    {context}
    </context>

    <question>
    {question}
    </question>

   {format_instructions}
"""

v006 = """Act as an expert in soccer scouting and player reports.
    Use the following information to provide a concise answer to the question enclosed in <question> tags.
    Dont make up anything that you dont see from the context.
    Every player should be included in your response. Make sure that for each unique player id from the context, there is an entry in your answer.

    <context>
    {context}
    </context>

    <question>
    {question}
    </question>

   {format_instructions}
"""

v007 = """You assist me in scouting soccer players.
    Dont make up anything that you dont see from the context.
    Make sure you follow the format instructions i give you in a json format and include EVERY player that I provide.
    So each unique player ID that is provided in the context absouletly has to be in your answer.
    
    <context>
    {context}
    </context>

    <question>
    {question}
    </question>

   {format_instructions}
"""

v008 = """
    Given the context and the question, create a summary of each unique player ID that you have in the <context> tags.

    <context>
    {context}
    </context>

    <question>
    {question}
    </question>

   {format_instructions}
"""

v009 = """
    <context>
    {context}
    </context>

    <question>
    {question}
    </question>

   {format_instructions}
"""





