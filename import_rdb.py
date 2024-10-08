# Import reports into Postgres to show them in the frontend after we summarized the reports
import argparse
import json

import psycopg2
import requests
from psycopg2 import sql
import sys
import urllib.request
from tqdm import tqdm

# Argument parser to accept a file name from the command line
parser = argparse.ArgumentParser()
parser.add_argument("--file", default="reports_test.json", nargs="?",
                    help="What file name to import from /data directory (default: reports_test.json)")

# Parse the command line arguments, with a default file if not provided
args, unknown = parser.parse_known_args()
import_file = "data/" + args.file

# SQL command to create a table if it does not exist already
sql_create_table = """CREATE TABLE IF NOT EXISTS report(
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            player_transfermarkt_id INTEGER NOT NULL,
            report TEXT NOT NULL)
            """

# Setup connection to the PostgreSQL database
conn = psycopg2.connect(database = "reports",
                        user = "postgres",
                        host= 'localhost',
                        password = "admin",
                        port = 5432)
conn.autocommit = True
print("Database connected successfully")

# Create a cursor object to execute SQL commands
cur = conn.cursor()
cur.execute(sql_create_table)
print("Table created successfully")

# Check if there are existing entries in the 'report' table
cur.execute('SELECT COUNT(*) FROM report;')
results = cur.fetchone()
if results[0] > 0:
    print("entries already exists, exiting")
    sys.exit() # Exit if there are already entries in the table

# Load the JSON data from the file specified by the user
with open(import_file, 'r') as file:
    data = json.load(file)
print("Data loaded successfully")

headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
}

# Fetch the player's name from Transfermarkt API using the player's ID
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

# Loop over the JSON data, inserting each entry into the database
for entry in tqdm(data, desc="Inserting data"):
    text = entry.get('text')
    player_tm_id = int(entry.get('player_transfermarkt_id'))
    name = get_name_from_tm(player_tm_id)

    sql = "INSERT INTO report (name, player_transfermarkt_id, report) VALUES (%s, %s, %s);"
    values = (name, player_tm_id, text)

    cur.execute(sql, values)

# Close the database cursor and connection
cur.close()
conn.close()

print("Data inserted successfully.")