from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

feature_cols = ['release_date', 'publisher', 'median_playtime', 'price', 'Genre: Action', 
                'Genre: Adventure', 'Genre: Casual', 'Genre: Early Access', 
                'Genre: Free to Play', 'Genre: Indie', 'Genre: Massively Multiplayer', 
                'Genre: RPG', 'Genre: Racing', 'Genre: Simulation', 'Genre: Sports', 'Genre: Strategy']

mymodel = joblib.load("mymodel.joblib")

@app.route('/infer', methods=['POST'])
def infer():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No input data provided"}), 400

    try:
        input_df = pd.DataFrame([data])[feature_cols]
    except KeyError as e:
        return jsonify({"error": f"Missing or incorrect features: {e}"}), 400

    infer_result = mymodel.predict(input_df)[0]

    response_data = {
        "result": {
            "value": infer_result
        }
    }

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
