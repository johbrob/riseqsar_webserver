import subprocess

from flask import Flask, request, render_template, url_for, flash, redirect, session
from flask.json import jsonify
import os
import sys

import argparse
sys.path.append(os.getcwd())
from plotting import create_plot
from forms import create_SmilesPredictForm
import requests

TEMPLATE_FOLDER = '/templates'
STATIC_FOLDER = '/static'
UPLOAD_FOLDER = '/uploads'

#app = Flask(__name__, template_folder=TEMPLATE_FOLDER, static_folder=STATIC_FOLDER, )
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000
app.config['SECRET_KEY'] = 'bfa7f83bbb83053cbccc82017295e703'

available_predictors = {}  # all available models with endpoints as keys


@app.route("/", methods=['GET', 'POST'])
@app.route("/home/", methods=['GET', 'POST'])
def home():
    form = create_SmilesPredictForm(available_predictors.keys())
    if request.method == 'POST':

        # update model choices
        available_endpoint_predictors = available_predictors[form.endpoint.data]
        form.model.choices = [(p['idx'], p['idx']) for p in available_endpoint_predictors]

        if form.errors:
            flash(form.errors)

        if form.validate_on_submit():
            predictor = available_endpoint_predictors[int(form.model.data)]

            session['prediction'] = {'smiles': form.smiles.data,
                                     'endpoint': form.endpoint.data,
                                     'predictor': predictor['name']}

            flash(f'{form.endpoint.data} property predicted for {form.smiles.data} using {predictor["name"]}',
                  'success')

            request_args = {'smiles': form.smiles.data,
                            'endpoint': form.endpoint.data,
                            'predictor_idx': int(form.model.data)}
            responce = requests.get(predictor['ip_address'] + '/predict', request_args)
            print(str(responce))
            session['prediction']['preds'] = responce.json()['preds']
            return redirect(url_for('home'))

    if 'prediction' in session:
        prediction = session.pop('prediction')
        plot = create_plot(prediction.pop('preds'))
        return render_template('molpredict.html', form=form, plot=plot, prediction=prediction)
    return render_template('molpredict.html', form=form)


@app.route("/model/<endpoint>")
def model(endpoint):
    endpoint_model_list = [{key: predictor[key] for key in ['name', 'idx']} for predictor in
                           available_predictors[endpoint]]
    return jsonify({'models': endpoint_model_list})


@app.route("/about")
def about():
    return render_template('about.html', title='About')


def merge_predictor_dicts(*pred_dicts_and_ips):
    """

    :param pred_dicts_and_ips: arbitrary large set of predictor dicts and corresponding ips as
            (pred_dicts1, ip1), (pred_dicts2, ip2), ...
    :return:
    """
    all_predictors = {}
    for (predictor_dict, ip) in pred_dicts_and_ips:
        for endpoint, predictors in predictor_dict.items():
            if endpoint not in all_predictors:
                all_predictors[endpoint] = []

            for predictor in predictors:
                predictor['ip_address'] = ip
                all_predictors[endpoint].append(predictor)

    return all_predictors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5004)
    parser.add_argument('--debug', type=bool, default=True)
    args = parser.parse_args()

    BASE = "http://0.0.0.0:3000"

    responce = requests.get(BASE + '/available_predictors')
    print(str(responce))
    available_torch_predictors = responce.json()
    available_other_predictors = {'hERG': [{'name': 'OtherPredictor 2022-08-01-T12.12.12', 'idx': 0}]}

    available_predictors = merge_predictor_dicts((available_torch_predictors, BASE),
                                                 (available_other_predictors, "http://0.0.0.0:3001"))
    app.run(host=args.host, port=args.port, debug=args.debug)

    # responce = requests.get(BASE + '/predict', {'smiles': 'CCC', 'endpoint': 'hERG', 'predictor_idx': 0})
    # print(responce)
    # print(responce.json())

    # print(responce.json())

    # available_predictors = get_available_predictors(args.model_dir)
    # app.run(host=args.host, port=args.port, debug=args.debug)
