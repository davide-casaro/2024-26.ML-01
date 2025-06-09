from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/infer', methods=['POST'])
def hello():
    data = request.get_json()
    param1 = data.get('param1')

    # mymodel = joblib.load("artefatto")
    # infer_result = mymodel.predict(param1)

    response_data = {
        "result": {
            "value": infer_result
        }
    }

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)