import json
import os
import sys

root = os.sep + os.sep.join(__file__.split(os.sep)[1 : __file__.split(os.sep).index("Drag-and-Drop-LLMs") + 1])
sys.path.append(root)
os.chdir(root)
os.environ["NUM_PROCESSES"] = "1"
# model_type = os.path.basename(__file__).split("_")[0]
model_type = "qwen7lora"

import torch
from fire import Fire
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from workspace.dnd.dataset import Text2Qwen25LoRA_FullCondDataset as Dataset
from workspace.dnd.model import HyperConvDecoderModel_SuperLarge as Model
from workspace.dnd.tokenizer import Qwen257BLoRA_Tokenizer2D as Tokenizer

SEED = 999
DATASET_ROOT = "./data/coding7B"
CONFIG_ROOT = "./workspace/datasets/coding7B"
COND_ROOT = "./prepare/data"
extractor = "./models/all-MiniLM-L12-v2"

import torch

torch.set_float32_matmul_precision("high")
import accelerate.utils
from torch.utils.data import DataLoader

accelerate.utils.set_seed(SEED)
max_text_length = 26624
modified_length = 1000
dataset_tag = "Code-74k-ShareGPT"
config: dict[str, [float, int, str, dict]] = {
    # global setting
    "need_test": False,
    # data setting
    "token_size": (16, 512),
    "real_length": 20,
    "num_texts": 16,
    "criterion_weight": torch.load(
        f"{CONFIG_ROOT}/{dataset_tag}/criterion_weight.pt", map_location="cpu", weights_only=True
    ),
    "extractor_type": "BERT",
    "text_tokenizer": AutoTokenizer.from_pretrained(extractor),
    "extra_condition_module": AutoModel.from_pretrained(extractor, torch_dtype="auto"),
    "max_text_length": max_text_length,
    "modified_length": modified_length,
    "model_config": {
        "features": [
            (16, modified_length, 384),
            (64, 500, 384),
            (256, 125, 400),
            (512, 64, 400),
            (1024, 64, 400),
            (2048, 32, 400),
            (4928, 16, 512),
        ],
        "condition_dim": (16, modified_length, 384),
        "kernel_size": 11,
    },
}


generate_config = {
    "device": "cuda",
    "num_generated": 20,
    "need_test": False,
}
config.update(generate_config)

# generate
print("==> Defining generate..")


def generate(model, loader, dataset, dstag_T, dstag_V):
    print("==> Generating...")
    model.eval()
    # prepare data
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    avg_time = []
    for idx, (tokens, cond_id, cond_mask, tag) in enumerate(loader):
        # generate
        if idx >= config["num_generated"]:
            break
        with torch.no_grad() and torch.autocast("cuda", dtype=torch.bfloat16):
            mask = ~torch.isnan(tokens)
            tokens = torch.nan_to_num_(tokens, nan=0.0)
            conditions = {
                "input_ids": cond_id.to(device=config["device"]),
                "attention_mask": cond_mask.to(device=config["device"]),
            }
            start.record()
            predict = model(
                source=None,
                mask=mask.to(config["device"]),
                condition=conditions,
                target=None,
                generate=True,
            )  # generate
            # save and log
            end.record()
            torch.cuda.synchronize()
            time = start.elapsed_time(end)
            if idx > 10:
                avg_time.append(time)
                print(f"generation_time {idx}: {time}")
    print(f"avg generation time:{sum(avg_time)/len(avg_time)/1000}s")


def main(eval_dataset: str, test_dataset: str):
    # Model
    print("==> Building model..")
    model = Model(
        config=config["model_config"],
        criterion_weight=config["criterion_weight"].view(1, -1, 1, 1),
        max_length=config["max_text_length"],
        modified_length=config["modified_length"],
        extractor_type=config["extractor_type"],
        extra_condition_module=config["extra_condition_module"],
    )
    tokenizer = Tokenizer(token_size=config["token_size"])

    diction = torch.load(f"./checkpoints/{model_type}__{eval_dataset}.pth", weights_only=True, map_location="cpu")
    model.load_state_dict(diction, strict=False)
    model.to(config["device"])

    # load module
    test_set = Dataset(
        checkpoint_folders=[f"{DATASET_ROOT}/{test_dataset}"],
        tokenizer=tokenizer,
        expected_iteration=None,
        real_length=config["real_length"],
        texts=[json.load(open(f"{COND_ROOT}/{test_dataset}.json", "r", encoding="utf-8"))],
        num_texts=config["num_texts"],
        text_tokenizer=config["text_tokenizer"],
        max_text_length=config["max_text_length"],
    )  # test_set
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=1,
        num_workers=0,
        collate_fn=test_set.collate_fn_test,
        shuffle=False,
    )  # test dataloader

    print(f"\n\nGenerating parameters for {test_dataset}")
    generate(model, test_loader, test_set, eval_dataset, test_dataset)


if __name__ == "__main__":
    Fire(main)
