from flask import Flask, request, jsonify
from flask_cors import cross_origin, CORS
from chain_summaries import invoke_summary_chain, invoke_single_report_chain
from rdb_access import fetch_reports_from_rdbms, all_player_ids_from_rdbms, all_players_with_name_from_rdbms
from reaction import log_reaction
from player_network import get_similar_players

app = Flask(__name__)
CORS(app)

@app.route("/scout-prompt", methods=["GET", "POST"])
@cross_origin()
def scout_prompt():
    if not request.is_json:
        return "Not a valid json!", 400

    user_query = request.get_json()['query']
    position = request.get_json()['position']
    fine_grained = request.get_json()['fineGrained']
    prompt_response = nlp_proccessing(user_query, position, fine_grained)
    return jsonify({"query": user_query, "response": prompt_response}), 200

@app.route("/reaction", methods=["POST"])
@cross_origin()
def reaction():
    if not request.is_json:
        return "Not a valid json!", 400

    log_reaction(request.get_json())
    return jsonify("message", "accepted"), 202

@app.route("/original-reports/<int:player_id>", methods=["GET"])
@cross_origin()
def original_reports(player_id):
    reports = fetch_reports_from_rdbms(player_id)
    return jsonify(reports), 200

@app.route("/playerids", methods=["GET"])
@cross_origin()
def player_ids():
    player_ids = all_player_ids_from_rdbms()
    return jsonify(player_ids), 200

@app.route("/players-with-name", methods=["GET"])
@cross_origin()
def players_with_names():
    players_with_names = all_players_with_name_from_rdbms()
    return jsonify(players_with_names), 200

@app.route("/similar_players/<int:player_id>", methods=["GET"])
@cross_origin()
def similar_players(player_id):
    try:
        similar_players = get_similar_players(player_id)
        return jsonify(similar_players), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 404

def nlp_proccessing(query, position, fine_grained):
    if fine_grained:
        return invoke_single_report_chain(query, position)
    else:
        return invoke_summary_chain(query, position)

if __name__ == "__main__":
    app.run()
