from flask import Flask, request, jsonify
from flask_cors import cross_origin, CORS
from chain_summaries import invoke_summary_chain, invoke_single_report_chain, llm_compare_players, get_summary_for_player_id
from model_definitions import ComparePlayerRequestPayload
from rdb_access import fetch_reports_from_rdbms, all_player_ids_from_rdbms, all_players_with_name_from_rdbms
from reaction import log_reaction
from player_network import get_similar_players

app = Flask(__name__)
CORS(app)

@app.route("/scout-prompt", methods=["GET", "POST"])
@cross_origin()
def scout_prompt():
    """Endpoint that takes the scouting form from the backend and returns a list of players suitable for this"""
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
    """Endpoint that logs the reaction to a log file"""
    if not request.is_json:
        return "Not a valid json!", 400

    log_reaction(request.get_json())
    return jsonify("message", "accepted"), 202

@app.route("/original-reports/<int:player_id>", methods=["GET"])
@cross_origin()
def original_reports(player_id):
    """Endpoint that returns the original reports of a player"""
    reports = fetch_reports_from_rdbms(player_id)
    return jsonify(reports), 200

@app.route("/playerids", methods=["GET"])
@cross_origin()
def player_ids():
    """Endpoint that returns all player ids we have in the relational database"""
    player_ids = all_player_ids_from_rdbms()
    return jsonify(player_ids), 200

@app.route("/players-with-name", methods=["GET"])
@cross_origin()
def players_with_names():
    """Endpoint that returns the all player names we have in the relational database"""
    players_with_names = all_players_with_name_from_rdbms()
    return jsonify(players_with_names), 200

@app.route("/similar_players/<int:player_id>", methods=["GET"])
@cross_origin()
def similar_players(player_id):
    """Endpoint that returns most similar player with their id and a similarity distance to the sent player_id"""
    similar_players = get_similar_players(player_id)
    return jsonify(similar_players), 200

@app.route("/compare-players", methods=["POST"])
@cross_origin()
def compare_players():
    """Endpoint that returns a comparison between players based on what criteria are sent in the incoming JSON"""
    request_raw_json = request.get_json()
    # parse to object
    comparePlayersPayload = ComparePlayerRequestPayload(**request_raw_json)
    # give to function
    comparison = llm_compare_players(comparePlayersPayload)
    comparison_response_payload = {"player_left": comparison.player_left,
                                   "player_left_name": comparison.player_left_name,
                                   "player_right": comparison.player_right,
                                   "player_right_name": comparison.player_right_name,
                                   "comparison": comparison.comparison}
    return jsonify(comparison_response_payload), 200

@app.route("/summary/<int:player_id>", methods=["GET"])
@cross_origin()
def summary(player_id):
    """Endpoint that returns summary of a player"""
    summary = get_summary_for_player_id(player_id)
    return jsonify(summary), 200

def nlp_proccessing(query, position, fine_grained):
    """Invokes the correct chain based on fine grained or not"""
    if fine_grained:
        return invoke_single_report_chain(query, position)
    else:
        return invoke_summary_chain(query, position)

if __name__ == "__main__":
    app.run()
