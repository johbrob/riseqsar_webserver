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
    predictor_addreses = ['http://riseqsar_webserver-herg_ogura_random_forest-1:5000']

    predictors = [] # list of tuples containing request responces and the corresponding addresses

    for address in predictor_addreses:
       responce = requests.get(address + '/available_predictors')
       predictors.append((responce.json(), address)) # append responce and address

    return merge_predictor_dicts(*predictors)
