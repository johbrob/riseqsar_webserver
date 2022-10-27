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


def _check_for_predictor(address):
    print(address + '/available_predictors')
    responce = requests.get(address + '/available_predictors')
    print(responce)
    return responce.json()

def available_predictors():
    available_ports = ['http://tmp_docker_test-herg_ogura_feed_forward_network-1:5000', 'http://tmp_docker_test-herg_ogura_random_forest-1:5000']

    all_predictors = [(_check_for_predictor(port), port) for port in available_ports]
    [print(str(predictor[0]), predictor[1]) for predictor in all_predictors]
    return merge_predictor_dicts(*all_predictors)
