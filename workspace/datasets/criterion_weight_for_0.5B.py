import os
import sys

import torch
from tqdm.auto import tqdm

root = os.sep + os.sep.join(__file__.split(os.sep)[1 : __file__.split(os.sep).index("Drag-and-Drop-LLMs") + 1])
sys.path.append(root)
os.chdir(root)

from workspace.dnd.dataset import CheckpointDataset_PaddedToSame as Dataset
from workspace.dnd.tokenizer import Qwen2505LoRA_Tokenizer2D as Tokenizer

datasets = ["ARC-c", "ARC-e", "OBQA", "BoolQ", "WinoGrande", "PIQA", "HellaSwag"]
DATASET_ROOT = f"./data/common_sense_reasoning"
CONFIG_ROOT = "./workspace/datasets/common_sense_reasoning"
for dataset_tag in datasets:
    Dataset.dtype = torch.float32

    dataset = Dataset(
        tokenizer=Tokenizer(token_size=(10, 130)),
        checkpoint_folder=f"{DATASET_ROOT}/{dataset_tag}",
        expected_iteration=None,
        real_length=50,
    )

    mean = torch.zeros((4296, 10, 130))
    now_numbers = 0
    for i in tqdm(range(len(dataset))):
        tokens = dataset[i][0]
        mean = (mean * now_numbers + tokens) / (now_numbers + 1)
        now_numbers += 1
    del tokens

    var = torch.zeros((4296, 10, 130))
    now_numbers = 0
    for i in tqdm(range(len(dataset))):
        tokens = dataset[i][0]
        tokens = torch.square(tokens - mean)
        var = (var * now_numbers + tokens) / (now_numbers + 1)
        now_numbers += 1

    std = torch.sqrt_(var)
    total_num = torch.sum(torch.where(std.isnan(), 0, 1))
    normed_std = std / std.abs().nansum()
    normed_std = torch.nanmean(normed_std, dim=[-1, -2])
    normed_std = normed_std * total_num
    normed_std = normed_std.flatten().detach().cpu()
    normed_std = normed_std.to(torch.float32)

    os.makedirs(f"{CONFIG_ROOT}/{dataset_tag}", exist_ok=True)
    torch.save(normed_std, f"{CONFIG_ROOT}/{dataset_tag}/criterion_weight.pt")
    print("total_parameters:", total_num)
    print("normed_std_values:", normed_std)
