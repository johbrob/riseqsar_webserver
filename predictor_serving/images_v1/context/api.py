import os
import pathlib
import pprint
from flask import Flask, request, abort
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
    predictor_idx = fields.Integer(required=True,
                                   validate=[Range(min=0, error="Predictor index must be greater than 0")])


predict_args_schema = PredictArgsSchema()


class Predict(Resource):

    def get(self):
        errors = predict_args_schema.validate(request.args)
        if errors:
            abort(400, str(errors))

        # select predictor
        pprint.pprint(available_predictors)
        print(request.args['endpoint'], request.args['predictor_idx'])

        predictor = available_predictors[request.args['endpoint']][int(request.args['predictor_idx'])]

        print(predictor)

        from riseqsar.models.descriptor_based_predictor import DescriptorbasedPredictor
        if isinstance(predictor['models'][0]['model'], DescriptorbasedPredictor):
            # featurize with one process using featurizer of first model
            featurized_mol = predictor['models'][0]['model'].featurizer.featurize([request.args['smiles']]).values
            # make predictions (predictor output can look different but we want preds as floats)
            preds = [float(model['model'].predict_proba_featurized(featurized_mol).squeeze()) for model in
                     predictor['models']]
        else:
            preds = [float(model['model'].predict_proba(request.args['smiles']).squeeze()) for model in
                     predictor['models']]

        thresholded_preds = [0 if pred < model['threshold'] else 1 for model, pred in zip(predictor['models'], preds)]

        print(preds)
        print(thresholded_preds)

        return {'preds': preds, 'thresholded_preds': thresholded_preds}


api.add_resource(AvailablePredictors, '/available_predictors')
api.add_resource(Predict, '/predict')

# print(os.getcwd())
# print(pathlib.Path('.').is_dir())
# print(pathlib.Path('predictors').is_dir())
# print(pathlib.Path('images_v0/predictors').is_dir())
available_predictors, available_predictor_references = init('../../..')
pprint.pprint(available_predictor_references)

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')
