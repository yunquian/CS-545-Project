import torch


def save_model(path, model):
    torch.save(model.state_dict(), path)


def load_model(path, model_class, *args, **kwargs):
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(path))
    return model
