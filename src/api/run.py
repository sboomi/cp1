import os
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from dotenv import load_dotenv, find_dotenv
from joblib import load

app = Flask(__name__)
api = Api(app)


ML_MODEL = os.environ.get("MODEL_PATH")
TOKEN = os.environ.get("TOKEN")


class Welcome(Resource):
    def get(self):
        return jsonify({
                    "Message": ("Bonjour, ceci est la beta d'un "
                                "algorithm d'analyse de sentiment"),
                    "Status Code": 200
                })


class SentimentAnalysis(Resource):
    def post(self):
        postedData = request.get_json()

        # Checking if all fields are present
        set1 = {"token", "text"}

        res = set(postedData.keys())
        if set1 != res:
            missing_fields = ', '.join(set1.difference(res))
            return jsonify({
                    "Message": f"{missing_fields} missing",
                    "Status Code": 400
                })

        token = postedData['token']
        text = postedData['text']

        # Checking if token is the good one
        if token != TOKEN:
            return jsonify({
                    "Message": "Token Invalide",
                    "Status Code": 401
                })

        #
        clf_pipe = load(ML_MODEL)
        prediction = clf_pipe.predict([text])[0]
        prediction = "Positif" if prediction == 1 else "Négatif"
        return jsonify({
                    "text": text,
                    "prediction": prediction,
                    "Status Code": 200
            }
            )


api.add_resource(SentimentAnalysis, "/sentiment")
api.add_resource(Welcome, "/welcome")

if __name__ == "__main__":
    load_dotenv(find_dotenv())
    app.run(debug=False, host='0.0.0.0', port=8080) 


