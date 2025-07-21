from safetensors import safe_open
from safetensors.torch import save_file


def load_safetensors(p: str, map_location="cpu", dtype=None) -> dict:
    new_diction = {}
    with safe_open(p, framework="pt", device=map_location) as f:
        for k in f.keys():
            new_diction[k] = f.get_tensor(k) if dtype is None else f.get_tensor(k).to(dtype)
    return new_diction


def save_safetensors(diction: dict, p: str):
    save_file(diction, p)
