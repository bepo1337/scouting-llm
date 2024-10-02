import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from typing import Optional, List
import json
from dataclasses import dataclass, asdict
import validators

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from bs4 import BeautifulSoup

import prompt_templates

# TODO make with parameters
# TODO STRG F with "prod" before running it to make sure i dont delete old data

load_dotenv()
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
OPENAI_API_VERSION = os.getenv('OPENAI_API_VERSION')
import_file = "data/team_prod.json"
output_file = "data/all_players_structured_report_summary_with_example_prod.json"
model_name="gpt-4o"
llm = AzureChatOpenAI(openai_api_key=AZURE_OPENAI_API_KEY, deployment_name=model_name)

@dataclass
class Report:
    scout_id: str
    text: str
    player_id: str
    player_transfermarkt_id: str
    grade_rating: Optional[float]
    grade_potential: Optional[float]
    main_position: str
    played_position: str

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(self.to_dict(), default=str)


def json_to_reports() -> [Report]:
    with open(import_file) as f:
        data = json.load(f)
        reports = [Report(
            scout_id=item.get('scout_id'),
            text=item.get('text'),
            player_id=item.get('player_id'),
            player_transfermarkt_id=item.get('player_transfermarkt_id'),
            grade_rating=item.get('grade_rating') if item.get('grade_rating') is not None else 0.0,
            grade_potential=item.get('grade_potential') if item.get('grade_potential') is not None else 0.0,
            main_position=item.get('main_position'),
            played_position=item.get('played_position')
        ) for item in data]

    return reports



def valid_text(text: str) -> bool:
    if text == "":
        return False

    # TODO maybe store reports to look if they're all invalid or usable
    if bool(BeautifulSoup(text, "html.parser").find()):
        return False

    if len(text) < 10:
        return False

    if validators.url(text):
        return False

    if text.isnumeric():
        return False

    #TODO check if it sends data somewhere else or is local
    #if len(text) > 50 and detect(text) != "en":
     #   return False

    return True

reports = json_to_reports()
player_id_to_reports = {}

for report in reports:
    player_id = report.player_transfermarkt_id
    if not valid_text(report.text):
        continue
    if player_id in player_id_to_reports:
        player_reports = player_id_to_reports[player_id]
        player_reports.append(report)
        player_id_to_reports[player_id] = player_reports
    else:
        player_id_to_reports[player_id] = [report]


def get_summary_from_llm(reports):
    prompt = prompt_templates.PROMPT_SUMMARY_INTO_STRUCTURE
    reportCount = 1
    for report in reports:
        prompt += f"Report {reportCount}: {report.text}\n\n"
        reportCount += 1

    answer = llm.invoke(prompt).content
    return answer


processed_players = []
for id, reports in player_id_to_reports.items():
    scout_id = 0
    main_position = reports[0].main_position
    player_id_scoutastic = reports[0].player_id # we dont use this, this is a scoutastic ID
    played_position = reports[0].played_position
    grade_potentials = []
    grade_ratings = []
    for report in reports:
        grade_potentials.append(report.grade_potential)
        grade_ratings.append(report.grade_rating)

    grade_potential = sum(grade_potentials)/len(grade_potentials)
    grade_rating = sum(grade_ratings)/len(grade_ratings)

    # only do summary if we have more than 1 report for this player.
    summary = ""
    # if len(reports) > 1:
    summary = get_summary_from_llm(reports)
    # else:
    #     summary = reports[0].text

    summarized_report = Report(player_id=player_id_scoutastic,
                               text=summary,
                               scout_id=scout_id,
                               player_transfermarkt_id=id,
                               grade_rating=grade_rating,
                               grade_potential=grade_potential,
                               main_position=main_position,
                               played_position=played_position)

    processed_players.append(summarized_report)

def reports_to_json(reports: List[Report]):
    reports_dict = [report.to_dict() for report in reports]
    return json.dumps(reports_dict, default=str)

with open(output_file, 'w') as file:
    file.write(reports_to_json(processed_players))
    print(f"successfully written to {output_file}")
