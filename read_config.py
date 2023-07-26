import json
from easydict import EasyDict


json_file = 'config.json'

with open(json_file, 'r') as config_file:
    config_dict = json.load(config_file)
    # EasyDict allows to access dict values as attributes (works recursively).
    config = EasyDict(config_dict)

print(config)