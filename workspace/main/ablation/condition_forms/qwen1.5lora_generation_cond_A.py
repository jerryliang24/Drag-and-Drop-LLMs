import json
import os
import sys

root = os.sep + os.sep.join(__file__.split(os.sep)[1 : __file__.split(os.sep).index("Drag-and-Drop-LLMs") + 1])
sys.path.append(root)
os.chdir(root)
os.environ["NUM_PROCESSES"] = "1"
model_type = os.path.basename(__file__).split("_")[0]

import torch
from fire import Fire
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from workspace.dnd.dataset import Text2Qwen25LoRA_CondADataset as Dataset
from workspace.dnd.model import HyperConvDecoderModel_SuperLarge as Model
from workspace.dnd.tokenizer import Qwen2515LoRA_Tokenizer2D as Tokenizer

SEED = 999
DATASET_ROOT = "./data/math1.5B"
CONFIG_ROOT = "./workspace/datasets/math1.5B"
COND_ROOT = "./prepare/data"
SAVE_ROOT = "./generated/math1.5B"
extractor = "./models/all-MiniLM-L12-v2"

import torch

torch.set_float32_matmul_precision("high")
import accelerate.utils
from torch.utils.data import DataLoader

accelerate.utils.set_seed(SEED)
max_text_length = 6144
modified_length = 1000
num_texts = 32
dataset_tag = "Competition_Math"
config: dict[str, [float, int, str, dict]] = {
    # global setting
    "need_test": False,
    # data setting
    "token_size": (18, 258),
    "real_length": 10,
    "num_texts": num_texts,
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
            (num_texts, modified_length, 384),
            (64, 500, 300),
            (256, 500, 300),
            (512, 125, 300),
            (1024, 64, 256),
            (2048, 32, 256),
            (4508, 18, 258),
        ],
        "condition_dim": (num_texts, modified_length, 384),
        "kernel_size": 11,
    },
}
model = Model(
    config=config["model_config"],
    criterion_weight=config["criterion_weight"].view(1, -1, 1, 1),
    max_length=config["max_text_length"],
    modified_length=config["modified_length"],
    extractor_type=config["extractor_type"],
    extra_condition_module=config["extra_condition_module"],
)
tokenizer = Tokenizer(token_size=config["token_size"])


generate_config = {
    "device": "cuda",
    "num_generated": 10,
    "need_test": False,
}
config.update(generate_config)


# generate
print("==> Defining generate..")


def generate(model, loader, dataset, dstag_T, dstag_V):
    print("==> Generating...")
    model.eval()
    # prepare data
    for idx, (tokens, cond_id, cond_mask, tag) in enumerate(loader):
        # generate
        with torch.no_grad() and torch.autocast("cuda", dtype=torch.bfloat16):
            mask = ~torch.isnan(tokens)
            tokens = torch.nan_to_num_(tokens, nan=0.0)
            conditions = {
                "input_ids": cond_id.to(device=config["device"]),
                "attention_mask": cond_mask.to(device=config["device"]),
            }

            import threading
            import time

            start_time = time.time()
            stop_timer = threading.Event()

            def display_time():
                while not stop_timer.is_set():
                    elapsed_time = time.time() - start_time
                    print(f"Elapsed time: {elapsed_time:.3f} seconds", end="\r")
                    time.sleep(0.05)

            timer_thread = threading.Thread(target=display_time)
            timer_thread.start()

            predict = model(
                source=None,
                mask=mask.to(config["device"]),
                condition=conditions,
                target=None,
                generate=True,
            )  # generate

            stop_timer.set()  # 告诉计时线程停止
            timer_thread.join()
            print()

        print(f"generate the {idx} th...")
        # save and log
        generated_norm = torch.square(predict[mask]).mean().item()
        original_norm = torch.square(tokens[mask]).mean().item()
        print("generated_start:", predict.flatten()[0:5].tolist())
        print("original_start:", tokens.flatten()[0:5].tolist())
        print("generated_end:", predict[0, -1, 0, 0:5].tolist())
        print("original_end:", tokens[0, -1, 0, 0:5].tolist())
        print("generated_l2norm:", generated_norm)
        print("original_l2norm:", original_norm)
        torch.cuda.empty_cache()
        # noinspection PyTypeChecker save checkpoint
        save_path = dataset.save_checkpoint(
            save_path=f"{SAVE_ROOT}/{dstag_T}T_on_{dstag_V}V", tokens=predict[0], tag=tag, number=idx
        )  # save files


def main(eval_dataset: str, test_dataset: str):
    # Model
    print("==> Building model..")
    diction = torch.load(f"./checkpoints/{model_type}__{eval_dataset}.pth", weights_only=True, map_location="cpu")
    model.load_state_dict(diction, strict=False)
    model.to(config["device"])

    # load module
    test_set = Dataset(
        checkpoint_folders=[f"{DATASET_ROOT}/Competition_Math"],
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
    generate(test_loader, test_set, eval_dataset, test_dataset)


if __name__ == "__main__":
    Fire(main)
