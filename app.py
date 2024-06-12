from flask import Flask, request, jsonify
import embed

embedding_model = embed.MistralEmbedding()
# llm = llm.
app = Flask(__name__)

@app.route("/scout-prompt", methods=["GET", "POST"])
def scout_prompt():
    if request.is_json:
        data = request.get_json()
        prompt_response = nlp_proccessing(data)
        return jsonify({"received": True, "data": data, "response": prompt_response}), 200

    return "Not a valid json!", 400


def nlp_proccessing(json_data):
    embedded_query = embedding_model.embed_query(json_data)
    # pass json data to our NLP stack
    print(embedded_query)
    return "Here's a list of players I advise to take a closer look: Bellingham, Jovic, Ekitike"

if __name__ == "__main__":
    app.run()
