import yaml
from easydict import EasyDict as edict

def get_config(cfg_yaml_path):
    with open(cfg_yaml_path) as f:
        cfg = edict(yaml.load(f))
    return cfg