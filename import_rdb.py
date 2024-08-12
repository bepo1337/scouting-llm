# Import reports into Postgres to show them in the frontend after we summarized the reports
import json

import psycopg2
from psycopg2 import sql
import sys


sql_create_table = """CREATE TABLE IF NOT EXISTS report(
            id SERIAL PRIMARY KEY,
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
for entry in data:
    text = entry.get('text')
    player_tm_id = int(entry.get('player_transfermarkt_id'))

    sql = "INSERT INTO report (player_transfermarkt_id, report) VALUES (%s, %s);"
    values = (player_tm_id, text)

    cur.execute(sql, values)

cur.close()
conn.close()

print("Data inserted successfully.")