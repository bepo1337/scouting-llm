import json
from dataclasses import dataclass, asdict

@dataclass
class Report:
    scout_id: str
    text: str
    player_id: str
    player_transfermarkt_id: str
    grade_rating: float
    grade_potential: float
    main_position: str
    played_position: str

def validate_report(report: dict) -> bool:
    """Validate the data types of a report."""
    try:
        assert isinstance(report['scout_id'], str), f"scout_id must be str, got {type(report['scout_id'])}"
        assert isinstance(report['text'], str), f"text must be str, got {type(report['text'])}"
        assert isinstance(report['player_id'], str), f"player_id must be str, got {type(report['player_id'])}"
        assert isinstance(report['player_transfermarkt_id'], str), f"player_transfermarkt_id must be str, got {type(report['player_transfermarkt_id'])}"
        assert isinstance(report['grade_rating'], float), f"grade_rating must be float, got {type(report['grade_rating'])}"
        assert isinstance(report['grade_potential'], float), f"grade_potential must be float, got {type(report['grade_potential'])}"
        assert isinstance(report['main_position'], str), f"main_position must be str, got {type(report['main_position'])}"
        assert isinstance(report['played_position'], str), f"played_position must be str, got {type(report['played_position'])}"
        return True
    except AssertionError as e:
        print(f"Validation error: {e}")
        return False

def json_to_reports(file_path: str) -> [Report]:
    with open(file_path) as f:
        data = json.load(f)
        valid_reports = []
        invalid_reports = []
        for item in data:
            if validate_report(item):
                valid_reports.append(Report(**item))
            else:
                invalid_reports.append(item)
        return valid_reports, invalid_reports

def write_invalid_reports_to_file(invalid_reports: list, file_path: str):
    with open(file_path, 'w') as f:
        for i, report in enumerate(invalid_reports, start=1):
            f.write(f"{i}: {report}\n")

def main():
    file_path = "data/uhh_data_prod.json"  # Update this path as needed
    valid_reports, invalid_reports = json_to_reports(file_path)
    
    print(f"Total valid reports: {len(valid_reports)}")
    print(f"Total invalid reports: {len(invalid_reports)}")

    if invalid_reports:
        invalid_reports_file = "data/invalid_reports.txt"
        write_invalid_reports_to_file(invalid_reports, invalid_reports_file)
        print(f"Invalid reports have been written to {invalid_reports_file}")

if __name__ == "__main__":
    main()
