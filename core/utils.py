# core/utils.py
import os
import yaml
import numpy as np

def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # expand seeds if given as "0-349"
    seeds = cfg["train"]["seeds"]
    if isinstance(seeds, str) and "-" in seeds:
        l, r = seeds.split("-")
        seeds = list(range(int(l), int(r) + 1))
    cfg["train"]["seeds"] = seeds
    return cfg

def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def set_global_seeds(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
