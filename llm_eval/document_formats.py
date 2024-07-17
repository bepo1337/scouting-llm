from collections import defaultdict

from langchain_core.documents import Document

def format_documents_v01(docs: [Document]):
    # Create a dictionary to hold reports for each player ID
    player_reports = defaultdict(list)

    # Aggregate reports by player ID
    for doc in docs:
        player_id = doc.metadata['player_transfermarkt_id']
        report_content = doc.page_content
        player_reports[player_id].append(report_content)

    # Format the aggregated reports
    formatted_reports = []
    for player_id, reports in player_reports.items():
        formatted_report = f"Player ID: {player_id}\n"
        for i, report in enumerate(reports, 1):
            formatted_report += f"Report {i}: {report}\n"
        formatted_report += "###"
        formatted_reports.append(formatted_report.strip())

    # Join all formatted reports into a single string
    return_string = "\n\n".join(formatted_reports)
    #print("------------\nAfter merging reports for each player:\n")
    #print(return_string)
    return return_string