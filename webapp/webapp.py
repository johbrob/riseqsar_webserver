from flask import Flask, request, render_template, url_for, flash, redirect, session
from flask.json import jsonify
import os
import sys
from pathlib import Path
import argparse
sys.path.append(os.getcwd())
from plotting import create_plot
from forms import create_SmilesPredictForm
import urllib.request


TEMPLATE_FOLDER = '/templates'
STATIC_FOLDER = '/static'
UPLOAD_FOLDER = '/uploads'

app = Flask(__name__, template_folder=TEMPLATE_FOLDER, static_folder=STATIC_FOLDER, )
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000
app.config['SECRET_KEY'] = 'bfa7f83bbb83053cbccc82017295e703'

available_predictors = {}     # all available models with endpoints as keys


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

            flash(f'{form.endpoint.data} property predicted for {form.smiles.data} using {predictor["name"]}', 'success')

            session['prediction']['preds'] = [model.predict_proba(form.smiles.data) for model in predictor['models']]
            # session['prediction']['preds'] = [0.9 + i/100 for i, model in enumerate(predictor['models'])]
            return redirect(url_for('home'))

    if 'prediction' in session:
        prediction = session.pop('prediction')
        plot = create_plot(prediction.pop('preds'))
        return render_template('molpredict.html', form=form, plot=plot, prediction=prediction)
    return render_template('molpredict.html', form=form)


@app.route("/model/<endpoint>")
def model(endpoint):
    endpoint_model_list = [{key: predictor[key] for key in ['name', 'idx']} for predictor in available_predictors[endpoint]]
    return jsonify({'models': endpoint_model_list})


@app.route("/about")
def about():
    return render_template('about.html', title='About')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=Path, required=True)
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--debug', type=bool, default=True)
    args = parser.parse_args()

    available_predictors = get_available_predictors(args.model_dir)
    app.run(host=args.host, port=args.port, debug=args.debug)
