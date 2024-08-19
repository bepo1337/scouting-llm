# Import reports into Postgres to show them in the frontend after we summarized the reports
import json

import psycopg2
import requests
from psycopg2 import sql
import sys
import urllib.request


sql_create_table = """CREATE TABLE IF NOT EXISTS report(
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            player_transfermarkt_id INTEGER NOT NULL,
            report TEXT NOT NULL)
            """

# setup connection
conn = psycopg2.connect(database = "reports",
                        user = "postgres",
                        host= 'localhost',
                        password = "admin",
                        port = 5432)
conn.autocommit = True

cur = conn.cursor()
cur.execute(sql_create_table)

# check if table entries exist
#   if exists --> exit script
cur.execute('SELECT COUNT(*) FROM report;')
results = cur.fetchone()
if results[0] > 0:
    print("entries already exists, exiting")
    sys.exit()


# run creation script TODO change here for other import path
json_file_path = "data/team_prod.json"
with open(json_file_path, 'r') as file:
    data = json.load(file)
# entry, add a report in the rdbms with
    # each report: tm_player_id, text

headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
}
def get_name_from_tm(player_transfermarkt_id) -> int:
    path = "https://www.transfermarkt.de/api/get/appShortinfo/player?ids=" + str(player_transfermarkt_id)
    contents = requests.get(path, headers=headers)

    jsonObj = json.loads(contents.text)
    try:
        name = jsonObj['player'][0]['name']
    except:
        print(f"error when fetching for player_id: {player_transfermarkt_id}")
        return ""
    return name

for entry in data:
    text = entry.get('text')
    player_tm_id = int(entry.get('player_transfermarkt_id'))
    name = get_name_from_tm(player_tm_id)

    sql = "INSERT INTO report (name, player_transfermarkt_id, report) VALUES (%s, %s, %s);"
    values = (name, player_tm_id, text)

    cur.execute(sql, values)

cur.close()
conn.close()

print("Data inserted successfully.")