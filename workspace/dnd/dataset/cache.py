import os
import random

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor


class CacheTensor:
    FORCE_RECACHE = False

    def __init__(self, tensor: Tensor, save_path: str, *args):
        self.total_size = tensor.size(0)
        self.save_path = save_path
        self.args = args
        self.indices = list(range(self.total_size))
        if (not os.path.exists(save_path)) or self.FORCE_RECACHE:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            diction = {str(index): slice for index, slice in enumerate(tensor)}
            save_file(diction, save_path)

    def random_get(self, number):
        random.shuffle(self.indices)
        this_indices = self.indices[:number]
        with safe_open(self.save_path, framework="pytorch", device="cpu") as f:
            tensors = [f.get_tensor(str(i)) for i in this_indices]
            tensors = torch.stack(tensors, dim=0)
        return tensors, torch.tensor(this_indices, dtype=torch.long), *self.args

    def sequential_get(self):
        with safe_open(self.save_path, framework="pytorch", device="cpu") as f:
            tensors = [f.get_tensor(str(i)) for i in range(len(self.indices))]
            tensors = torch.stack(tensors, dim=0)
        return tensors, *self.args
