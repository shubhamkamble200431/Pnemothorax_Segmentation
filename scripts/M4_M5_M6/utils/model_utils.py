import os
import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model):
    temp_path = "temp_model.pth"
    torch.save(model.state_dict(), temp_path)
    size = os.path.getsize(temp_path) / (1024 * 1024)
    os.remove(temp_path)
    return size