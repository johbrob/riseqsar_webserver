from flask import Flask, session
#from predictor_loading import get_available_predictors

app = Flask(__name__)


@app.route("/init")
def init():
    #load all models
    #session['available_models'] = get_available_predictors
    #session['initialized'] = True
    pass


@app.route("/available_models")
def available_models():
    # gives info about all available models
    #if not session['initialized']:
    #    init()

    #return session['available_models']
    pass

@app.route("/predict")
def predict():
    # gives smiles and which model and endpoint to use for making prediction
    
    # how pass parameters?
    pass

if __name__ == "__main__":
    app.run()
    #session['initialized'] = False