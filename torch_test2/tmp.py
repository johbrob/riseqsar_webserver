import requests

# BASE = "http://127.0.0.1:5001/"
BASE = "http://0.0.0.0:3000/"

# responce = requests.get(BASE + '/predict', {'smiles': 'CCC',
#                                              'endpoint': 'hERG',
#                                              'predictor_idx': 0}, timeout=30)
# print(responce)
# print(responce.json())

responce = requests.get(BASE + '/argpass', {'smiles': 'CCC', 'endpoint': 'hERG', 'predictor_idx': 0})
print(responce.json())

responce = requests.get(BASE + '/modelpass')
print(responce.json())