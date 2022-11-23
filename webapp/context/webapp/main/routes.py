from flask import Blueprint
from flask import request, render_template, url_for, flash, redirect
from flask.json import jsonify
from flask_login import login_required
from .utils import create_plot, smiles_2_b64_img, available_predictors
from .forms import create_SmilesPredictForm
import requests
import statistics


main = Blueprint('main', __name__)


predictors = available_predictors()



@main.route("/", methods=['GET', 'POST'])
@main.route("/home/", methods=['GET', 'POST'])
# @login_required
def home():
    return redirect(url_for('main.predict'))

@main.route("/predict", methods=['GET', 'POST'])
@main.route("/predict/<smiles>/<property_endpoint>/<predictor_idx>", methods=['GET', 'POST'])
def predict(smiles=None, property_endpoint=None, predictor_idx=None):
    form = create_SmilesPredictForm(predictors.keys())

    if request.method == 'POST':
        # update model choices
        endpoint_predictors = predictors[form.endpoint.data]
        form.model.choices = [(i, i) for i in range(len(endpoint_predictors))]
        if form.errors:
            flash(form.errors)

        if form.validate_on_submit():
            predictor = endpoint_predictors[int(form.model.data)]

            flash(f'{form.endpoint.data} property predicted for {form.smiles.data} using {predictor["name"]}',
                  'success')

            return redirect(url_for('main.predict',
                                    smiles=form.smiles.data,
                                    property_endpoint=form.endpoint.data,
                                    predictor_idx=form.model.data))

    if smiles and property_endpoint and predictor_idx:
        predictor = predictors[property_endpoint][int(predictor_idx)]
        responce = requests.get(url=predictor['ip_address'] + '/predict',
                                params={'smiles': smiles,
                                        'endpoint': property_endpoint,
                                        'predictor_idx': predictor['idx']})
        preds = responce.json()['preds']
        thresholded_preds = responce.json()['thresholded_preds']
        avg_thresholded_preds = statistics.mean(thresholded_preds)

        mol_img = smiles_2_b64_img(smiles)
        #draw_n_save_mol(smiles)
        # pred_plot = create_plot(preds)

        return render_template('predict.html',
                               title='Predict',
                               form=form,
                               # plot=pred_plot,
                               mol_img=mol_img,
                               prediction={'smiles': smiles,
                                           'endpoint': property_endpoint,
                                           'preds': preds,
                                           'thresholded_preds': str(avg_thresholded_preds),
                                           # 'pred_plot': pred_plot,
                                           'predictor': predictor['name']})
    else:
        return render_template('predict.html', form=form)


@main.route("/model/<endpoint>")
def model(endpoint):
    endpoint_model_list = [{key: predictor[key] for key in ['name', 'idx']} for predictor in predictors[endpoint]]
    return jsonify({'models': endpoint_model_list})


@main.route("/about")
def about():
    return render_template('about.html', title='About')
