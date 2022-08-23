import requests

BASE = "http://127.0.0.1:5001/"

responce = requests.get(BASE + '/predict', {'smiles': 'CCC',
                                             'endpoint': 'hERG',
                                             'predictor_idx': 0}, timeout=30)
print(responce.json())