
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from joblib import dump, load

app = Flask(__name__)
api = Api(app)

class Welcome(Resource):
    def get(self):
        return jsonify({
                    "Message" : "Bonjour, ceci est la beta d'un algorithm d'analyse de sentiment",
                    "Status Code": 200
                })
        
class SentimentAnalysis(Resource):
    def post(self):
        postedData = request.get_json()

        #checking if all fields are present
        set1 = {"token", "text"}

        res = set(postedData.keys())
        if set1 != res:
            missing_fields = ', '.join(set1.difference(res))
            return jsonify({
                    "Message" : f"{missing_fields} missing",
                    "Status Code": 400
                })

        token = postedData['token']
        text = postedData['text']

        #checking if token is the good one
        if token != "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9":
            return jsonify({
                    "Message" : "Token Invalide",
                    "Status Code": 401
                })

        #
        clf_pipe = load('sentiment_pipe.joblib')
        prediction = clf_pipe.predict([text])[0]
        prediction = "Positif" if prediction == 1 else "Négatif"
        return jsonify({
                    "text" : text,
                    "prediction" : prediction,
                    "Status Code": 200
            }
            )

api.add_resource(SentimentAnalysis, "/sentiment")
api.add_resource(Welcome, "/welcome")

if __name__=="__main__":
    app.run(debug=False, host='0.0.0.0', port=8080) 


