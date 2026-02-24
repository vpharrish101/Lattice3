import yaml

def load_cfg(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)