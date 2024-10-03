import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument("--importfile", default="reports_test.json", nargs="?",
                    help="What file name to import from /data directory (default: reports_test.json)")
parser.add_argument("--outputfile", default="summaries_test.json", nargs="?",
                    help="What file name to output to /data directory (default: summaries_test.json)")


args, unknown = parser.parse_known_args()

import_file = "data/" + args.importfile
output_file = "data/" + args.outputfile

load_dotenv()
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
OPENAI_API_VERSION = os.getenv('OPENAI_API_VERSION')
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
    """Loads the JSON file and returns a list of reports"""
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
    """Validates the report text and returns false if the text is empty, text is html, text is less than 10 characters, is an URL or if it is numeric"""
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

# Go through all reports and sort them by player ID
# player_id_to_reports will then be a map of player ids to a list of reports
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
    """Invokes the LLM to get a summary for the reports"""
    prompt = prompt_templates.PROMPT_SUMMARY_INTO_STRUCTURE
    reportCount = 1
    for report in reports:
        prompt += f"Report {reportCount}: {report.text}\n\n"
        reportCount += 1

    answer = llm.invoke(prompt).content
    return answer


# Now we go through the previously created map and create a summary based on those reports
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

    summary = ""
    summary = get_summary_from_llm(reports)

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
    """Parses the reports to a JSON list"""
    reports_dict = [report.to_dict() for report in reports]
    return json.dumps(reports_dict, default=str)

with open(output_file, 'w') as file:
    file.write(reports_to_json(processed_players))
    print(f"successfully written to {output_file}")
