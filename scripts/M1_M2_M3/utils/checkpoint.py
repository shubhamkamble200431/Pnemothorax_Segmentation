import torch


def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, device):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model