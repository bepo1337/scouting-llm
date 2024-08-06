from flask import Flask, request, jsonify
from flask_cors import cross_origin
from flask_cors import CORS

from chain_summaries import invoke_chain
from reaction import log_reaction

app = Flask(__name__)
CORS(app)

@app.route("/scout-prompt", methods=["GET", "POST"])
@cross_origin()
def scout_prompt():
    if not request.is_json:
        return "Not a valid json!", 400

    user_query = request.get_json()['query']
    prompt_response = nlp_proccessing(user_query)
    return jsonify({"query": user_query, "response": prompt_response}), 200

@app.route("/reaction", methods=["POST"])
@cross_origin()
def reaction():
    if not request.is_json:
        return "Not a valid json!", 400

    log_reaction(request.get_json())
    return jsonify("message", "accepted"), 202




def nlp_proccessing(query):
    return invoke_chain(query)

if __name__ == "__main__":
    app.run()
