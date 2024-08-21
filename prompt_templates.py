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
###EXAMPLE###
"
**General text about the player:**
He is an attacking midfielder who, despite experiencing a slight decline in his physical abilities, remains a valuable asset due to his game organization and strategic composure. His ability to scan situations and manage the game enhances various phases of play. His commitment and experience could be valuable, especially when strategically integrated within the team and salary constraints of the MLS. He is expected to perform competently for at least one or two more years in the league.

**Strengths:**
- Excellent game organization and situational scanning
- Strategic composure and effective game management
- Courage and willingness to track back defensively
- Experience and commitment

**Weaknesses:**
- Physical decline affecting overall performance
- High salary demands potentially not justifiable
- Defensive functions may not align perfectly with his profile

**Physical Capabilities:**
- Slight decline in physical ability
- Still reasonably dynamic for his age

**Offensive Capabilities:**
- Precise in various phases of the game
- Ability to make a difference and contribute strategically

**Defensive Capabilities:**
- Willingness to track back and help defensively
- Courageous in defensive situations despite physical decline

**Other Attributions:**
- Needs careful management of physical condition and interventions
- Integration within the team, especially alongside players like XYZ needs careful consideration to avoid exposure
"
###END OF EXAMPLE###

Do not make anything up that you dont see from the reports. You can also leave parts empty if the reports dont say anything about them. But still put in the headlines.

The reports about the player are the following:
"""

PROMPT_QUERY_INTO_STRUCTURED_QUERY_WITH_EXAMPLE = """
Context is soccer and you are a assistant in scouting. I want you to restructure the following query into a structured format.
The structure for query is as follows:
- General text about the player
- Strengths
- Weaknesses
- Physical capabilities
- Offensive capabilities
- Defensive capabilities
- Other attributions

This is an example for a query and a corresponding structured format I expect you to return, but yours can vary in length and content depending on the query in the context:

###EXAMPLE###
Query: "A dynamic offensive midfielder with excellent technical skills, sharp vision, and creative playmaking abilities.
He consistently finds and executes key passes, never gives up, and leads by example with his relentless work rate.
Physically, he is agile, strong, and possesses great stamina, enabling him to maintain high intensity throughout the match.
Key player in breaking down defenses and supporting the team with high energy, intelligence, and resilience."

Structured format:
"
**General text about the player:**
A dynamic offensive midfielder with excellent technical skills, sharp vision, and creative playmaking abilities. 

**Strengths:**
- Consistently finds and executes key passes.
- Never gives up and leads by example with his relentless work rate.

**Weaknesses:**
- Not specified in the query

**Physical Capabilities:**
- Agile, strong, and possesses great stamina.
- Capable of maintaining high intensity throughout the match.

**Offensive Capabilities:**
- Key player in breaking down defenses.

**Defensive Capabilities:**
- Not specified 

**Other Attributions:**
- Supports the team with high energy, intelligence, and resilience.
"
###END OF EXAMPLE###

Do not make anything up that you dont see from the query.
You can also leave parts empty if the query doesnt say anything about them. But still put in the headlines.

The query I want you to restructure is the following:
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


Do not make anything up that you dont see from the reports.

The reports about the player are the following:

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





