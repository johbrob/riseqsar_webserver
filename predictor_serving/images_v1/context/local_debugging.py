from pathlib import Path
from predictor_loading import get_available_predictors
import numpy as np

if __name__ == '__main__':
    dir = Path('./predictors/random_forest/herg_ogura')
    a = get_available_predictors(dir)
    print(a)
    predictor = a['hERG'][0]

    # featurize with one process using featurizer of first model
    featurized_mol = predictor['models'][0].featurizer.featurize(['CCC']).values

    # make predictions
    preds = [float(model.predict_proba_featurized(featurized_mol).squeeze()) for model in predictor['models']]
    print(type(preds[0]))
    for pred in preds:
        print(pred)

    # check types
    # a = np.array(1.0)
    # a = a.squeeze()
    # print(type(a))
    # print(float(a))
    # print(type(float(a)))