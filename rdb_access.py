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
