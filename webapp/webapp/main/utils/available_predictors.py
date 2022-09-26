import requests

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

def available_predictors():
    BASE = "http://0.0.0.0:3001"
    responce = requests.get(BASE + '/available_predictors')
    print(str(responce))
    available_torch_predictors = responce.json()
    available_other_predictors = {'hERG': [{'name': 'OtherPredictor 2022-08-01-T12.12.12', 'idx': 0}]}
    return merge_predictor_dicts((available_torch_predictors, BASE),
                                                 (available_other_predictors, "http://0.0.0.0:3001"))
