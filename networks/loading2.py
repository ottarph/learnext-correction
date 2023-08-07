import torch
import torch.nn as nn
from pathlib import Path
import json
import yaml

from typing import Literal
from os import PathLike

from networks.loading import ModelBuilder

def build_model(model_dict: dict) -> nn.Module:

    assert len(model_dict.keys()) == 1
    key = next(iter(model_dict.keys()))
    val = model_dict[key]

    model: nn.Module = getattr(ModelBuilder, key)(val)

    return model


def load_model(model_dir: PathLike, load_state_dict: bool = True,
                            mode: Literal["yaml", "json"] = "yaml") -> nn.Module:
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise ValueError("Non-existent directory.")

    if mode == "yaml":
        with open(model_dir / "model.yml", "r") as infile:
            model_dict = yaml.safe_load(infile.read())
    elif mode == "json":
        with open(model_dir / "model.json", "r") as infile:
            model_dict= json.loads(infile.read())
    else:
        raise ValueError
    model = build_model(model_dict)

    if load_state_dict:
        model.load_state_dict(torch.load(model_dir / "state_dict.pt"))

    return model


if __name__ == "__main__":
    import json
    with open("networks/test/test_model/model.json", "r") as infile:
        obj = json.loads(infile.read())

    print(obj)

    model = build_model(obj)
    print(model)

    model = load_model("networks/test/test_model")
    print(model)
