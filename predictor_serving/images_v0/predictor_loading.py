import pickle
from pathlib import Path
import sys
from riseqsar.experiment.experiment_tracker import find_experiment_top_level_models


def _get_this_python_env():
    return Path(sys.prefix).name   # I am not sure if one shuld use sys.prefix or sys.base_prefix


def _load_model(path_to_pickled_model):
    with open(path_to_pickled_model, 'rb') as f:
        model_dict = pickle.load(f)
    deserializer = model_dict['model_factory']
    model_bytes = model_dict['model_state']

    return deserializer(model_bytes)


def _get_available_models_of_this_predictor(dir_of_this_predictor):
    all_available_predictor_models = []

    smiles_to_test_on = 'Cc1ccccc1'
    featurized_mol = None

    for model_dir in find_experiment_top_level_models(
            dir_of_this_predictor):  # don't know if this is functions intended use
        path_to_pickled_model = model_dir / Path('models/final_model.pkl')

        try:  # because might load models using "wrong" conda environment or might load non-functioning models
            model = _load_model(path_to_pickled_model)  # might fail here with ImportError...
            assert hasattr(model, 'predict_proba'), f"{type(model)} has no attribute 'predict_proba'"  # ...or here if predict_proba doesn't exist...
            assert hasattr(model, 'featurizer'), f"{type(model)} has no attribute 'featurizer'"
            assert callable(getattr(type(model), 'predict_proba')), f"{type(model)} has no callable 'predict_proba'"
            assert callable(getattr(type(model), 'predict_proba_featurized')), f"{type(model)} has no callable 'predict_proba'"

            if featurized_mol is None:
                featurized_mol = model.featurizer.featurize([smiles_to_test_on]).values
            model.predict_proba_featurized(featurized_mol)
            # model.predict_proba('Cc1ccccc1')  # ...or here if predict_proba is not implemented
            all_available_predictor_models.append(model)

        except (ImportError, AssertionError, NotImplementedError, FileNotFoundError) as e:
            print(e)

    return all_available_predictor_models


def _get_predictor_info(models, path_to_predictor_dir, endpoint, available_predictors):
    predictor_info = {'models': models,
                      'path': path_to_predictor_dir,
                      'class': type(models[0]),
                      'idx': 0}

    predictor_class_as_string = str(type(models[0])).split("'")[1].split(".")[-1]
    # predictor_info['name'] = predictor_class_as_string + ' ' + predictor_info['path'].parent.name
    predictor_info['name'] = predictor_class_as_string

    if endpoint in available_predictors:
        predictor_info['idx'] = len(available_predictors[endpoint])

    return predictor_info


def get_available_predictors(dir_of_all_predictors):
    available_predictors = {}
    if isinstance(dir_of_all_predictors, str):
        dir_of_all_predictors = Path(dir_of_all_predictors)
    # look through all dataset_spect.pkl files in model_dir
    for exp_spec_file in dir_of_all_predictors.rglob('*/experiment_specification.pkl'):
        exp_config_file = exp_spec_file.parent / Path('experiment_config.pkl')
        with open(exp_config_file, 'rb') as f:  # get endpoint form dataset_spec
            exp_config = pickle.load(f)
        endpoint = exp_config.dataset_spec_collection.dataset_specs[0].dataset_endpoint

        all_available_models = _get_available_models_of_this_predictor(exp_spec_file.parent.parent)

        if all_available_models:
            predictor_info = _get_predictor_info(all_available_models,
                                                 exp_spec_file.parent.parent,
                                                 endpoint,
                                                 available_predictors)

            if endpoint in available_predictors:
                available_predictors[endpoint].append(predictor_info)
            else:
                available_predictors[endpoint] = [predictor_info]

    return available_predictors

if __name__ == '__main__':
    # dir = Path('/home/johbro/PycharmProjects/rise-qsar/models')
    dir = Path('predictors')
    a = get_available_predictors(dir)
    print(a)