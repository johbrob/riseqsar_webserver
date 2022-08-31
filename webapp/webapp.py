import subprocess

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
import docker
import time


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


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


def spin_up_container(username, password, container_img, container_name):
    # docker
    client = docker.from_env()
    client.login(username=username, password=password, registry='https://registry-1.docker.io/')
    img = client.images.pull(container_img) # make sure we have image

    containers = client.containers.list()
    for container in containers:
        if container.name == container_name:
            user_permits_delete_container = query_yes_no(f'A container with name {container_name} already exists. Do you wanna remove it to run this one?')

            if user_permits_delete_container:
                container.stop()
                container.remove()

            else:
                print(f'Will try use existing container {container_name}')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('docker_username', type=str)
    parser.add_argument('docker_password', type=str)
    #parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--debug', type=bool, default=True)
    args = parser.parse_args()

    torch_container_name = 'webapp_torch_model_server'
    torch_img = 'johbrob/rq-torch-env'

    import os
    #spin_up_container(args.docker_username, args.docker_password, torch_img, torch_container_name)
    #print('starting executing subprocess line from ', os.getpid())
    # subprocess.call(f"python run_container.py '{args.docker_username}' '{args.docker_password}'", shell=True)
    #a = subprocess.check_output(f"python run_container.py '{args.docker_username}' '{args.docker_password}'", shell=True)
    #print(a)
    #print('finishing executing subprocess line from', os.getpid())

    import requests
    time.sleep(0.1)
    #BASE = "http://0.0.0.0:3000"
    #try:
    #    responce = requests.get(BASE + '/available_predictors')
    #    print(responce)
    #    print(responce.json())
    #except requests.exceptions.ConnectionError as e:
    #    print(e)
    #    print('could not send request')
    #    container.stop()
    #    container.remove()

    BASE = "http://0.0.0.0:3000"

    responce = requests.get(BASE + '/predict', {'smiles': 'CCC', 'endpoint': 'hERG', 'predictor_idx': 0})
    print(responce)
    print(responce.json())

    responce = requests.get(BASE + '/available_predictors')
    print(responce.json())

    #available_predictors = get_available_predictors(args.model_dir)
    #app.run(host=args.host, port=args.port, debug=args.debug)
