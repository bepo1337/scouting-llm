# Relational Database Access
import psycopg2

conn = psycopg2.connect(database = "reports",
                        user = "postgres",
                        host= 'localhost',
                        password = "admin",
                        port = 5432)

cursor = conn.cursor()
def fetch_reports_from_rdbms(playerID: int):
    sql = "SELECT report FROM report WHERE player_transfermarkt_id = (%s);"
    values = (playerID,)

    cursor.execute(sql, values)
    results = cursor.fetchall()
    return [row[0] for row in results]

def fetch_name_from_rdbms(playerID: int):
    sql = "SELECT name FROM report WHERE player_transfermarkt_id = (%s);"
    values = (playerID,)

    cursor.execute(sql, values)
    result = cursor.fetchone()
    if result:
        return result[0]
    else:
        return None

def all_player_ids_from_rdbms():
    sql = "SELECT player_transfermarkt_id FROM report;"

    cursor.execute(sql)
    results = cursor.fetchall()
    return [row[0] for row in results]


def all_players_with_name_from_rdbms():
    sql = "SELECT player_transfermarkt_id, name FROM report;"

    cursor.execute(sql)
    results = cursor.fetchall()
    return [{"id": row[0], "name": row[1]} for row in results]