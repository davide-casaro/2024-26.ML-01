from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

feature_cols=['release_date', 'publisher', 'median_playtime', 'price', 'Genre: Action', 
              'Genre: Adventure', 'Genre: Casual', 'Genre: Early Access', 
              'Genre: Free to Play', 'Genre: Indie', 'Genre: Massively Multiplayer', 
              'Genre: RPG', 'Genre: Racing', 'Genre: Simulation', 'Genre: Sports', 'Genre: Strategy']

@app.route('/infer', methods=['POST'])
def hello():
    data = request.get_json()
    param1 = data.get("param1", "")
    
    mymodel = joblib.load("mymodel.joblib")
    infer_result = mymodel.predict(param1)

    response_data = {
        "result": {
            "value": infer_result
        }
    }

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)