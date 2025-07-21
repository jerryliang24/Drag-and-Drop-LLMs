import os

SEED = 999
import json
import os
import sys

root = os.sep + os.sep.join(__file__.split(os.sep)[1 : __file__.split(os.sep).index("Drag-and-Drop-LLMs") + 1])
sys.path.append(root)
os.chdir(root)
with open("./workspace/main/config.json", "r") as f:
    workspace_config = json.load(f)
USE_WANDB = workspace_config["use_wandb"]
if USE_WANDB:
    import wandb

    os.environ["WANDB_API_KEY"] = workspace_config["wandb_api_key"]

# torch
import torch

torch.set_float32_matmul_precision("high")
import accelerate.utils
from accelerate import Accelerator
from bitsandbytes.optim import AdamW8bit as Optimizer
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

accelerate.utils.set_seed(SEED)

DATASET_ROOT = "./data/math1.5B"
CONFIG_ROOT = "./workspace/datasets/math1.5B"
COND_ROOT = "./prepare/data"

from workspace.dnd.dataset import Text2Qwen25LoRA_FullCondDataset as Dataset
from workspace.dnd.model import HyperConvDecoderMdel_SuperLarge as Model
from workspace.dnd.module import WarmupScheduler
from workspace.dnd.tokenizer import Qwen2515LoRA_Tokenizer2D as Tokenizer
from workspace.dnd.tools import calculate_mean_criterion_weight, start_monitor

datasets = ["Competition_Math", "Math_QA", "Math-IIO-68K-Mini", "Math-Plus", "Mu-Math", "ToT-Math-V1"]
dataset_tag = "Math1.5B"
extractor = "./models/all-MiniLM-L12-v2"
accelerator = Accelerator()
max_text_length = 4608
modified_length = 1000
num_texts = 32
config: dict[str, [float, int, str, dict]] = {
    # global setting
    "seed": SEED,
    "model_tag": os.path.basename(__file__)[:-3].split("_")[1],
    "need_test": False,
    "use_wandb": True,
    # data setting
    "token_size": (16, 256),
    "real_length": 80,
    "train_checkpoint_folders": [f"{DATASET_ROOT}/{dataset}" for dataset in datasets],
    "test_checkpoint_folder": "",
    "dataset_tag": dataset_tag,
    "generated_file": f"{CONFIG_ROOT}/{dataset_tag}/",
    # train setting
    "max_num_gpus": 8,
    "batch_size": 128,
    "num_workers": 2,
    "prefetch_factor": 1,
    "warmup_steps": 1,
    "total_steps": 5000,
    "learning_rate": 3e-5,
    "weight_decay": 0.1,
    "max_grad_norm": 1.0,
    "save_every": 100,
    "print_every": 20,
    "num_texts": num_texts,
    "save_folder": "./checkpoints",
    "noise_enhance": 0.0001,
    "criterion_weight": calculate_mean_criterion_weight(
        [f"{CONFIG_ROOT}/{dataset}/criterion_weight.pt" for dataset in datasets]
    ),
    "extractor_type": "BERT",
    "text_tokenizer": AutoTokenizer.from_pretrained(extractor),
    "extra_condition_module": AutoModel.from_pretrained(extractor, torch_dtype="auto").to(accelerator.device),
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
            (4508, 16, 256),
        ],
        "condition_dim": (num_texts, modified_length, 384),
        "kernel_size": 11,
    },
}

# Data
if accelerator.is_main_process:
    print("==> Preparing tokenizer and dataset...")
tokenizer = Tokenizer(token_size=config["token_size"])
Dataset.dtype = torch.bfloat16
expected_iteration = config["total_steps"] * config["batch_size"] * config["max_num_gpus"]


train_set = Dataset(
    checkpoint_folders=config["train_checkpoint_folders"],
    tokenizer=tokenizer,
    expected_iteration=expected_iteration,
    num_texts=config["num_texts"],
    text_tokenizer=config["text_tokenizer"],
    max_text_length=config["max_text_length"],
    real_length=config["real_length"],
    texts=[json.load(open(f"{COND_ROOT}/{dataset}.json", "r", encoding="utf-8")) for dataset in datasets],
)  # train set


# test_set = Dataset(
#     checkpoint_folders=[f"{DATASET_ROOT}/Evol-Instruct-Code-80K-V1"],
#     tokenizer=tokenizer,
#     expected_iteration = expected_iteration,
#     num_texts=config["num_texts"],
#     text_tokenizer=config["text_tokenizer"],
#     max_text_length=config["max_text_length"],
#     real_length=config["real_length"],
#     texts=[json.load(open(f"{COND_ROOT}/Evol-Instruct-Code-80K-V1.json","r",encoding="utf-8"))],
# )


# process dataloader
config["batch_size"] = config["batch_size"] // int(os.environ["NUM_PROCESSES"])

if accelerator.is_main_process:
    print(f"batchsize:{config['batch_size']}; total:{config['batch_size'] * int(os.environ['NUM_PROCESSES'])}")
train_loader = DataLoader(
    dataset=train_set,
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    prefetch_factor=config["prefetch_factor"],
    collate_fn=train_set.collate_fn_train,
    drop_last=True,
    shuffle=True,
)  # train dataloader
eval_loader = DataLoader(
    dataset=train_set,
    batch_size=1,
    num_workers=0,
    collate_fn=train_set.collate_fn_test,
    shuffle=False,
)  # eval dataloader
eval_iterator = iter(eval_loader)
# test_loader = DataLoader(
#     dataset=test_set,
#     batch_size=1,
#     num_workers=0,
#     collate_fn=test_set.collate_fn_test,
#     shuffle=False,
# )  # test dataloader
# test_iterator = iter(test_loader)


model = Model(
    config=config["model_config"],
    criterion_weight=config["criterion_weight"].view(1, -1, 1, 1),
    max_length=config["max_text_length"],
    modified_length=config["modified_length"],
    extractor_type=config["extractor_type"],
    extra_condition_module=config["extra_condition_module"],
).to(accelerator.device)

# Optimizer
if accelerator.is_main_process:
    print("==> Building optimizer and scheduler...")
optimizer = Optimizer(
    params=model.parameters(),
    lr=config["learning_rate"],
    weight_decay=config["weight_decay"],
)  # use 8bit optimizer
scheduler = WarmupScheduler(
    optimizer=optimizer,
    warmup_steps=config["warmup_steps"],
    total_steps=config["total_steps"],
)  # use cosine annealing


# Accelerator
if __name__ == "__main__":
    # accelerator
    # from accelerate.utils import DistributedDataParallelKwargs
    # kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # accelerator = Accelerator(kwargs_handlers=[kwargs,],)

    # noinspection PyTypeChecker
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)  # define everything


# Wandb
config["tag"] = config["model_tag"] + "__" + config["dataset_tag"]
# noinspection PyUnboundLocalVariable
if __name__ == "__main__" and USE_WANDB and accelerator.is_main_process:
    wandb.login(key=workspace_config["wandb_api_key"])
    wandb.init(
        project="DnD",
        name=config["tag"],
        config=config,
    )
    start_monitor(second=20)


# Train
def train():
    if not USE_WANDB:
        train_loss = 0
        this_steps = 0
    if accelerator.is_main_process:
        print("==> Training...")
    model.train()
    for batch_idx, (tokens, cond_id, cond_mask) in enumerate(train_loader):
        conditions = {"input_ids": cond_id.to(accelerator.device), "attention_mask": cond_mask.to(accelerator.device)}
        optimizer.zero_grad()
        tokens = tokens.to(accelerator.device)
        mask = ~torch.isnan(tokens)
        tokens = torch.nan_to_num_(tokens, nan=0.0)
        # noinspection PyArgumentList
        with accelerator.autocast():
            loss = model(
                source=None,
                mask=mask,
                condition=conditions,
                target=tokens,
                noise_enhance=config.get("noise_enhance", None),
            )  # forward
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
        optimizer.step()
        if accelerator.is_main_process:
            scheduler.step(batch_idx)
            # log ans update
            if USE_WANDB:
                wandb.log(
                    {"train_loss": loss.item(), "learning_rate:": optimizer.state_dict()["param_groups"][0]["lr"]}
                )  # update diction
            else:  # not use wandb
                # noinspection PyUnboundLocalVariable
                train_loss, this_steps = train_loss + loss.item(), this_steps + 1
                if this_steps % config["print_every"] == 0:
                    if accelerator.is_main_process:
                        print(f"step:{this_steps} Loss: {train_loss/this_steps:.6f}")
                    train_loss = 0
            if batch_idx % config["save_every"] == 0:
                os.makedirs(config["save_folder"], exist_ok=True)
                state = accelerator.unwrap_model(model).state_dict()
                state = accelerator.unwrap_model(model).state_dict()
                keys_to_delete = [key for key in state.keys() if key.startswith("condition_module")]
                for key in keys_to_delete:
                    del state[key]
                # noinspection PyTypeChecker
                torch.save(state, os.path.join(config["save_folder"], config["tag"] + ".pth"))
                if batch_idx == 1000:
                    torch.save(state, os.path.join(config["save_folder"], config["tag"] + "1000.pth"))
                if batch_idx == 2000:
                    torch.save(state, os.path.join(config["save_folder"], config["tag"] + "2000.pth"))
                if batch_idx == 3000:
                    torch.save(state, os.path.join(config["save_folder"], config["tag"] + "3000.pth"))
                if batch_idx == 4000:
                    torch.save(state, os.path.join(config["save_folder"], config["tag"] + "4000.pth"))
                if batch_idx == 5000:
                    torch.save(state, os.path.join(config["save_folder"], config["tag"] + "5000.pth"))
                if accelerator.is_main_process:
                    print("\nEvaluating on eval set:")
                generate(iterator=eval_iterator, idx=batch_idx // config["save_every"])

                torch.cuda.empty_cache()
        if batch_idx >= config["total_steps"]:
            break


# Generate
def generate(iterator, idx):
    torch.cuda.empty_cache()
    if accelerator.is_main_process:
        print("==> Generating...")
    model.eval()
    # prepare data
    tokens, cond_id, cond_mask, tag = next(iterator)
    conditions = {"input_ids": cond_id.to(accelerator.device), "attention_mask": cond_mask.to(accelerator.device)}
    # generate
    with torch.no_grad() and torch.autocast("cuda", dtype=torch.bfloat16):
        mask = ~torch.isnan(tokens)
        tokens = torch.nan_to_num_(tokens, nan=0.0)
        predict = accelerator.unwrap_model(model).generate(
            source=None,
            mask=mask.to(accelerator.device),
            condition=conditions,
            target=None,
            # generate=True,
        )  # generate
    # save and log
    generated_norm = torch.square(predict[mask]).mean().item()
    original_norm = torch.square(tokens[mask]).mean().item()
    if accelerator.is_main_process:
        print("generated_start:", predict.flatten()[0:5].tolist())
        print("original_start:", tokens.flatten()[0:5].tolist())
        print("generated_end:", predict[0, -1, 0, 0:5].tolist())
        print("original_end:", tokens[0, -1, 0, 0:5].tolist())
        print("generated_l2norm:", generated_norm)
        print("original_l2norm:", original_norm)
    if USE_WANDB:
        wandb.log(
            {
                "generated_l2norm": generated_norm,
                "original_l2norm:": original_norm,
            }
        )  # log generated info
    # noinspection PyTypeChecker save checkpoint
    _ = train_set.save_checkpoint(
        save_path=config["generated_file"], tokens=predict[0], tag=tag, number=idx
    )  # save files
    torch.cuda.empty_cache()
    model.train()
    return predict


if __name__ == "__main__":
    train()
