import os
import yaml
import torch


def save_checkpoint(state, filename):
    print("=> Saving checkpoint")
    with open("config.yml") as config_file:
        config = yaml.safe_load(config_file)
    torch.save(state, os.path.join(config["checkpoint_path"], filename))


def load_checkpoint(filename, model, optimizer):
    print("=> Loading checkpoint")
    with open("config.yml") as config_file:
        config = yaml.safe_load(config_file)
    checkpoint = torch.load(os.path.join(config["checkpoint_path"], filename))
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step
