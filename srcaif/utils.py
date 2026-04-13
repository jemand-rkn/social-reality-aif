from typing import Iterable, Union
import torch.nn as nn
import torch
import numpy as np


def tensor2numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if x.device.type == "cpu":
        return x.detach().numpy()
    else:
        return x.cpu().detach().numpy()


def get_parameters(modules: Iterable[nn.Module]):
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters


def get_grad_norm(parameters: Union[torch.Tensor, Iterable[torch.Tensor]], norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        # prevent generators from being exhausted
        parameters = list(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = torch.nn.utils.clip_grad._get_total_norm(grads, norm_type)
    return total_norm
