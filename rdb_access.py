# Relational Database Access
import os
import psycopg2
rdbms_host = os.getenv("RDBMS_HOST", "localhost")

# Setup connection
conn = psycopg2.connect(database = "reports",
                        user = "postgres",
                        host= rdbms_host,
                        password = "admin",
                        port = 5432)

cursor = conn.cursor()

# Fetch reports for a single player from the relational database
def fetch_reports_from_rdbms(playerID: int):
    sql = "SELECT report FROM report WHERE player_transfermarkt_id = (%s);"
    values = (playerID,)

    cursor.execute(sql, values)
    results = cursor.fetchall()
    return [row[0] for row in results]

# Fetch name for a single player from the relational database
def fetch_name_from_rdbms(playerID: int):
    sql = "SELECT name FROM report WHERE player_transfermarkt_id = (%s);"
    values = (playerID,)

    cursor.execute(sql, values)
    result = cursor.fetchone()
    if result:
        return result[0]
    else:
        return None

# Fetch all player ids from the relational database
def all_player_ids_from_rdbms():
    sql = "SELECT player_transfermarkt_id FROM report;"

    cursor.execute(sql)
    results = cursor.fetchall()
    return [row[0] for row in results]

# Fetch all players including their name and their id from the relational database
def all_players_with_name_from_rdbms():
    sql = "SELECT DISTINCT player_transfermarkt_id, name FROM report;"

    cursor.execute(sql)
    results = cursor.fetchall()
    return [{"id": row[0], "name": row[1]} for row in results]