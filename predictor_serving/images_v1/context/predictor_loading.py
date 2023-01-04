from riseqsar.experiment.experiment_tracker import find_experiment_top_level_models
from riseqsar.models.descriptor_based_predictor import DescriptorbasedPredictor
from pathlib import Path
import itertools
import pickle
import csv
import sys

predictorClass2NameMap = {'RandomForestPredictor': 'Random Forest',
                          'DeepNeuralNetworkDescriptorbasedPredictor': 'Deep Neural Network',
                          'GraphDeepNeuralNetworkPredictor': 'Graph Neural Network'}


def _get_this_python_env():
    return Path(sys.prefix).name   # I am not sure if one shuld use sys.prefix or sys.base_prefix


def _load_model(path_to_pickled_model):
    with open(path_to_pickled_model, 'rb') as f:
        model_dict = pickle.load(f)
    deserializer = model_dict['model_factory']
    model_bytes = model_dict['model_state']

    return deserializer(model_bytes)

def _load_threshold(path_to_file):
    with open(path_to_file, newline='') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        threshold = float(next(itertools.islice(file_reader, 1, None))[1])
    if not threshold:
        raise ValueError('No threshold found in %s' % path_to_file)
    return threshold



def _get_available_models_of_this_predictor(dir_of_this_predictor):
    all_available_predictor_models = []
    all_available_predictor_model_thresholds = []

    smiles_to_test_on = 'Cc1ccccc1'
    featurized_mol = None

    for model_dir in find_experiment_top_level_models(
            dir_of_this_predictor):  # don't know if this is functions intended use
        path_to_pickled_model = model_dir / Path('models/final_model.pkl')
        path_to_threshold = model_dir / Path('threshold.csv')

        try:  # because might load models using "wrong" conda environment or might load non-functioning models
            model = _load_model(path_to_pickled_model)      # might fail here with ImportError...
            threshold = _load_threshold(path_to_threshold)  # could also throw some error here...check which

            # predictors should always have "predict_proba" method
            assert hasattr(model, 'predict_proba'), f"{type(model)} has no attribute 'predict_proba'"  # ...or here if predict_proba doesn't exist...
            assert callable(getattr(type(model), 'predict_proba')), f"{type(model)} has no callable 'predict_proba'"

            if isinstance(model, DescriptorbasedPredictor):
                # DescriptorBasedPredictors should have a featurizer and "predict_proba_featurized" method
                assert hasattr(model, 'featurizer'), f"{type(model)} has no attribute 'featurizer'"
                assert callable(getattr(type(model), 'predict_proba_featurized')), f"{type(model)} has no callable 'predict_proba'"

                # test DescriptorBasedPredictor
                if featurized_mol is None:
                    featurized_mol = model.featurizer.featurize([smiles_to_test_on]).values
                model.predict_proba_featurized(featurized_mol)
            else:
                # test other predictors that doesn't use featurizer
                model.predict_proba(smiles_to_test_on)
            # model.predict_proba('Cc1ccccc1')  # ...or here if predict_proba is not implemented
            all_available_predictor_models.append({'model': model, 'threshold': threshold})

        except (ImportError, AssertionError, NotImplementedError, FileNotFoundError) as e:
            print(e)

    return all_available_predictor_models


def _get_predictor_info(models, path_to_predictor_dir, endpoint, available_predictors):
    predictor_info = {'models': models,
                      'path': path_to_predictor_dir,
                      'class': type(models[0]),
                      'idx': 0}

    predictor_class_as_string = str(type(models[0]['model'])).split("'")[1].split(".")[-1]
    ensamble_info = f' (Ensamble of {len(models)} models)' if len(models) > 1 else ''
    predictor_info['name'] = predictorClass2NameMap[predictor_class_as_string] + ensamble_info

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
            predictor_info = _get_predictor_info(all_available_models, exp_spec_file.parent.parent,
                                                 endpoint, available_predictors)

            if endpoint in available_predictors:
                available_predictors[endpoint].append(predictor_info)
            else:
                available_predictors[endpoint] = [predictor_info]

    return available_predictors

if __name__ == '__main__':
    # dir = Path('/home/johbro/PycharmProjects/rise-qsar/models')
    _load_threshold('../predictors/ffn/herg_ogura/2022-09-27T09.09.36/resamples/resample_00/threshold.csv')
    # a = get_available_predictors(Path('.'))
    # print(a)