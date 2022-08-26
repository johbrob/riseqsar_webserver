from flask import Flask, session
from flask.json import jsonify
from flask_restful import Resource, Api, reqparse
from predictor_loading import get_available_predictors

app = Flask(__name__)
api = Api(app)

available_predictors = {}
available_predictor_references = {}

def create_external_rep(available_predictors):
    ext_rep = {}

    for endpoint in available_predictors:
        ext_rep[endpoint] = []
        for predictor in available_predictors[endpoint]:
            ext_rep[endpoint].append({'name': predictor['name'],
                                      'idx': predictor['idx']})
    return ext_rep

def init(dir):
    #load all models
    available_predictors = get_available_predictors(dir)
    available_predictor_references = create_external_rep(available_predictors)
    return available_predictors, available_predictor_references


class AvailablePredictors(Resource):
    def get(self):
        if not available_predictor_references:
            print('Somethihng is not right...')

        return available_predictor_references

predict_put_args = reqparse.RequestParser()
predict_put_args.add_argument('smiles', type=str, help='Molecule in SMILES format', required=True)
predict_put_args.add_argument('endpoint', type=str, help='Property endpoint which we want to predict', required=True)
predict_put_args.add_argument('predictor_idx', type=int, help='Index of predictor given in the predictor dictionary', required=True)

class Predict(Resource):

    def get(self):
        args = predict_put_args.parse_args()

        predictor = available_predictors[args.endpoint][args.predictor_idx]
        featurized_mol = predictor['models'][0].featurizer.featurize([args.smiles]).values
        preds = [model.predict_proba_featurized(featurized_mol)[0] for model in predictor['models']]
        print(preds)
        print(type(preds))
        return {'preds': preds}


api.add_resource(AvailablePredictors, '/available_predictors')
api.add_resource(Predict, '/predict')


if __name__ == "__main__":
    available_predictors, available_predictor_references = init('experiments')
    # app.run(host='0.0.0.0')
    app.run(debug=True, host='0.0.0.0')
