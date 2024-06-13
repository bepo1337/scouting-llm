from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

player_id_schema = ResponseSchema(name="player_id",
                                  description="ID of the player that the report is about")

report_summary_schema = ResponseSchema(name="report_summary",
                                  description="Summary of the report content about the player")

response_schemas = [player_id_schema, report_summary_schema]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instruction = output_parser.get_format_instructions()

print("format instrcution: ", format_instruction)
