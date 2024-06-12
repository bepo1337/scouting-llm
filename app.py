from flask import Flask, request, jsonify
import embed
from chain import invoke_chain

embedding_model = embed.NomicEmbedding()
# llm = llm.
app = Flask(__name__)

@app.route("/scout-prompt", methods=["GET", "POST"])
def scout_prompt():
    if request.is_json:
        data = request.get_json()
        prompt_response = nlp_proccessing(data)
        return jsonify({"received": True, "data": data, "response": prompt_response}), 200

    return "Not a valid json!", 400


def nlp_proccessing(query):
    # embedded_query = embedding_model.embed_query(json_data)
    # pass json data to our NLP stack
    # print(embedded_query)
    return invoke_chain(query)

if __name__ == "__main__":
    app.run()
