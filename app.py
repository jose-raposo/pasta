from flask import Flask, render_template, request
from flask_restful import reqparse, abort, Api, Resource
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# instantiating Flask RESTFul API
app = Flask(__name__)
api = Api(app)

# loading model
model = joblib.load('model.pkl')

# argument parsing
# features to serve the model
parser = reqparse.RequestParser()
parser.add_argument('idade', type=float)
parser.add_argument('esv', type = float, action='append')
parser.add_argument('essv', type = float, action='append')
parser.add_argument('imve', type = float, action='append')
parser.add_argument('vae', type = float, action='append')
parser.add_argument('u', type = float, action='append')
parser.add_argument('creat', type = float, action='append')
parser.add_argument('k', type = float, action='append')
parser.add_argument('ct', type = float, action='append')

class PredictProbability(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        try:
            idade = float(request.args.get('idade'))
            esv = float(request.args.get('esv'))
            essv = float(request.args.get('essv'))
            imve = float(request.args.get('imve'))
            vae = float(request.args.get('vae'))
            u = float(request.args.get('u'))
            creat = float(request.args.get('creat'))
            k = float(request.args.get('k'))
            ct = float(request.args.get('ct'))
        except:
            pass

    def post(self):
        try:
            pred_proba = model.predict_proba(np.array([idade, esv,
            essv, imve, vae, u,
            creat, k, ct]).reshape(-1, 9))[:,1]


            # round the predict proba value and set to new variable
            var = int(pred_proba*100)

            # create JSON object
            output = {'prediction': str(var)+'%'}
        
            return output
        except:
            return 'Request already not made'

# Setup the Api resource routing here
# Route the URL to the resource

api.add_resource(PredictProbability, '/')


if __name__ == '__main__':
    app.run(debug=True)