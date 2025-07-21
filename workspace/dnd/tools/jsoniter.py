import json
import os.path
import warnings
from pprint import pprint

import torch

from .iterator import File, Iterator


def calculate_mean_criterion_weight(weight_path: str):
    mean_weight = []
    for weight in weight_path:
        mean_weight.append(torch.load(weight, map_location="cpu", weights_only=True))
    mean_weight = torch.mean(torch.stack(mean_weight), dim=0)

    return mean_weight


class JsonFile(File):
    def __init__(self, path, read_only: bool = False):
        if not os.path.exists(path):
            self.create_on(path)
        self.read_only = read_only
        super().__init__(path=path)

    @property
    def diction(self):
        this_diction = {}
        with self as d:
            this_diction.update(d)
        return this_diction

    def show(self):
        pprint(self.diction)

    def __enter__(self) -> dict:
        with open(self.path, "r") as f:
            self.content = json.load(f)
        return self.content

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None and self.read_only is False:
            with open(self.path, "w") as f:
                json.dump(self.content, f, indent=2)
            del self.content
        return False

    @classmethod
    def create_on(cls, path: str):
        if os.path.exists(path):
            warnings.warn(f"{path} have been existed.")
        with open(path, "w") as f:
            f.write("{}")
        return JsonFile(path=path)


class JsonIterator(Iterator):
    def __init__(self, root, **kwargs):
        super().__init__(root=root, file_class=JsonFile, **kwargs)
