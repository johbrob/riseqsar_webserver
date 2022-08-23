import pickle
from pathlib import Path
from riseqsar.training.experiment import find_experiment_top_level_models

def _class_to_name(model):
    from riseqsar.models.logistic_regression import LogisticRegressionPredictor
    from riseqsar.models.random_forest import RandomForestPredictor
    from riseqsar.models.svm import SVMClassifier
    from riseqsar.models.xgboost import XGBoostPredictor
    from riseqsar.models.neural_networks.feedforward_network import DeepNeuralNetworkDescriptorbasedPredictor
    from riseqsar.models.neural_networks.graph_neural_network import GraphDeepNeuralNetworkPredictor
    from riseqsar.models.neural_networks.smiles_transformer import SmilesTransformerPredictor
    from riseqsar.models.neural_networks.recurrent_networks import RecurrentNetworkPredictor
    if isinstance(model, LogisticRegressionPredictor):
        return 'Logistic Regression'
    elif isinstance(model, RandomForestPredictor):
        return 'Random Forest'
    elif isinstance(model, SVMClassifier):
        return 'Support Vector Machine (SVM)'
    elif isinstance(model, DeepNeuralNetworkDescriptorbasedPredictor):
        return 'Deep Neural Network (DNN)'
    elif isinstance(model, XGBoostPredictor):
        return 'XGBoost'
    elif isinstance(model, GraphDeepNeuralNetworkPredictor):
        return 'Graph Neural Network (GNN)'
    elif isinstance(model, SmilesTransformerPredictor):
        return 'Transformer'
    elif isinstance(model, RecurrentNetworkPredictor):
        return 'Recurrent Neural Network (RNN)'
    else:
        return str(type(model)).split("'")[1].split(".")[-1]


def _load_model(path_to_pickled_model):
    with open(path_to_pickled_model, 'rb') as f:
        model_dict = pickle.load(f)
    deserializer = model_dict['model_factory']
    model_bytes = model_dict['model_state']

    return deserializer(model_bytes)


def _get_available_models_of_this_predictor(dir_of_this_predictor):
    all_available_predictor_models = []

    for model_dir in find_experiment_top_level_models(dir_of_this_predictor):  # don't know if this is functions intended use
        path_to_pickled_model = model_dir / Path('models/final_model.pkl')

        try:  # because might load models using "wrong" conda environment or might load non-functioning models
            model = _load_model(path_to_pickled_model)           # might fail here with ImportError...
            assert hasattr(type(model), 'predict_proba')        # ...or here if predict_proba doesn't exist...
            assert callable(getattr(type(model), 'predict_proba'))
            model.predict_proba('Cc1ccccc1')                  # ...or here if predict_proba is not implemented
            all_available_predictor_models.append(model)

        except (ImportError, AssertionError, NotImplementedError) as e:
            print(e)

    return all_available_predictor_models


def _get_predictor_info(models, path_to_predictor_dir, endpoint, available_predictors):
    predictor_info = {'models': models,
                      'path': path_to_predictor_dir,
                      'class': type(models[0]),
                      'idx': 0}

    predictor_class_as_string = str(type(models[0])).split("'")[1].split(".")[-1]
    predictor_info['name'] = predictor_class_as_string + ' ' + predictor_info['path'].parent.name

    if endpoint in available_predictors:
        predictor_info['idx'] = len(available_predictors[endpoint])

    return predictor_info


def get_available_predictors(dir_of_all_predictors):
    available_predictors = {}

    # look through all dataset_spect.pkl files in model_dir
    for dataset_specs_path in dir_of_all_predictors.rglob('*/dataset_specs.pkl'):

        with open(dataset_specs_path, 'rb') as f:   # get endpoint form dataset_spec
            data = pickle.load(f)
        endpoint = data.dataset_specs[0].dataset_endpoint

        all_available_models = _get_available_models_of_this_predictor(dataset_specs_path.parent.parent)

        if all_available_models:
            predictor_info = _get_predictor_info(all_available_models,
                                                 dataset_specs_path.parent.parent,
                                                 endpoint,
                                                 available_predictors)

            if endpoint in available_predictors:
                available_predictors[endpoint].append(predictor_info)
            else:
                available_predictors[endpoint] = [predictor_info]

    return available_predictors

if __name__ == '__main__':
    dir = Path('/home/johbro/PycharmProjects/rise-qsar/models')
    get_available_predictors(dir)