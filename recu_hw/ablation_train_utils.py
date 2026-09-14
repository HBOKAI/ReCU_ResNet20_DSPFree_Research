import math


def finetune_tau(config):
    return float(config["finetune"].get("tau", 0.99))
