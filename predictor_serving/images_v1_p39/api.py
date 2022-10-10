import json
import pprint
from flask import Flask, session, request, abort
from flask.json import jsonify
from flask_restful import Resource, Api
from marshmallow import Schema, fields
from marshmallow.validate import Range
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


def init(predictor_dir):
    # load all models
    available_predictors = get_available_predictors(predictor_dir)
    available_predictor_references = create_external_rep(available_predictors)
    return available_predictors, available_predictor_references


class AvailablePredictors(Resource):
    def get(self):
        if not available_predictor_references:
            print('Somethihng is not right...')

        return available_predictor_references


class PredictArgsSchema(Schema):
    smiles = fields.Str(required=True)
    endpoint = fields.Str(required=True)
    predictor_idx = fields.Integer(required=True, validate=[Range(min=0, error="Predictor index must be greater than 0")])


predict_args_schema = PredictArgsSchema()


class Predict(Resource):

    def get(self):
        errors = predict_args_schema.validate(request.args)
        if errors:
            abort(400, str(errors))

        # select predictor
        predictor = available_predictors[request.args['endpoint']][int(request.args['predictor_idx'])]

        # featurize with one process using featurizer of first model
        predictor['models'][0].featurizer.set_nproc(1)
        featurized_mol = predictor['models'][0].featurizer.featurize([request.args['smiles']]).values

        # make predictions
        preds = [model.predict_proba_featurized(featurized_mol)[0] for model in predictor['models']]
        return {'preds': preds}


api.add_resource(AvailablePredictors, '/available_predictors')
api.add_resource(Predict, '/predict')

if __name__ == "__main__":
    available_predictors, available_predictor_references = init('predictors')
    pprint.pprint(available_predictor_references)
    app.run(debug=True, host='0.0.0.0')
