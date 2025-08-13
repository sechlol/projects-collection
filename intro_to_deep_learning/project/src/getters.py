from typing import Dict

from torch import nn, optim

import common
import models


def get_optimizer(model, cmd_args: Dict):
    lr = float(cmd_args.get("-lr"))
    optimizers = {
        0: optim.AdamW(model.parameters(), lr=lr),
        1: optim.Adam(model.parameters(), lr=lr),
        2: optim.SGD(model.parameters(), lr=lr),
        3: optim.Adamax(model.parameters(), lr=lr),
        4: optim.Adadelta(model.parameters(), lr=lr),
    }
    return optimizers[cmd_args.get("-opt", 0)]


def get_loss_function(cmd_args: Dict):
    loss_functions = {
        0: nn.BCELoss(),
        1: nn.MultiLabelSoftMarginLoss(),
        2: nn.MSELoss(),
        3: nn.L1Loss(),
    }

    return loss_functions[cmd_args.get("-loss", 0)]


def get_model(cmd_args: Dict, device):
    m = cmd_args.get("-m")
    is_rgb = cmd_args.get("-rgb")
    num_classes = len(common.LABELS)
    default_model = models.Model0(num_classes=num_classes, is_rgb=is_rgb)

    if m == 3 and not is_rgb:
        raise ValueError("Model 3 (ModelResPreTrained) only supports RGB images")

    model_map = {
        0: default_model,
        1: models.Model1(num_classes, is_rgb),
        2: models.Model2(num_classes, is_rgb),
        3: models.ModelResPreTrained(num_classes),
        4: models.Dummy(num_classes, cmd_args),
        5: models.Model5(num_classes, is_rgb),
        6: models.Model5AvgPooling(num_classes, is_rgb),
    }

    return model_map.get(m, default_model).to(device)
