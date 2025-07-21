import os
import random
from collections import OrderedDict

import torch
from torch import Tensor

from workspace.dnd.tools import load_safetensors, save_safetensors

from .dataset import CheckpointDataset_PaddedToSame, DataLoadingError


class Text2Qwen25LoRA_CheckpointDataset(CheckpointDataset_PaddedToSame):
    dtype = torch.bfloat16

    def __init__(
        self,
        checkpoint_folder: str,
        tokenizer,
        expected_iteration: int,
        condition_path: str,
        num_texts: int = None,
        real_length: int = None,
    ):
        super().__init__(
            checkpoint_folder=checkpoint_folder,
            tokenizer=tokenizer,
            expected_iteration=expected_iteration,
            real_length=real_length,
        )
        self.condition = torch.load(condition_path, weights_only=True, map_location="cpu")
        self.num_texts = num_texts

    def __getitem__(self, item):
        try:  # If encounter some error, we return the next data.
            checkpoint_path = self.checkpoints[item % self.real_length]
            # condition = self.extract_condition(item % self.real_length)
            condition = self.condition
            diction = self.load_checkpoint(checkpoint_path)
            tokens, others = self.tokenizer.tokenize(diction, **self.tokenizer_extra_config)
        except DataLoadingError:
            return self[random.randint(0, self.real_length - 1)]
        return tokens, condition, checkpoint_path

    def extract_condition(self, idx: int) -> Tensor:
        # We have extracted conditions externally and we only need to load at this step.
        start = idx * self.num_texts
        end = (idx + 1) * self.num_texts
        assert (
            start >= 0 and end <= self.condition.shape[0]
        ), f"index{idx} out of bounds with {idx}*{self.num_texts}={idx*self.num_texts}"

        tensor = self.condition[start:end]
        # this is for the debugging
        # tensor = torch.randn([1,10,100,3584])
        return tensor

    @staticmethod
    def post_process(dict) -> dict:
        new_dict = {}
        for k, v in dict.items():
            k = k.replace("base_model.model.", "")
            new_dict[k] = v
        return new_dict

    def load_checkpoint(self, load_path: str) -> OrderedDict:
        diction = load_safetensors(load_path, map_location="cpu", dtype=self.dtype)
        diction = self.post_process(diction)
        diction = OrderedDict(sorted(diction.items(), key=lambda x: self.sort_key(x)))
        return diction

    def load_checkpoint_for_saving(self, load_path: str) -> OrderedDict:
        diction = load_safetensors(load_path, map_location="cpu", dtype=self.dtype)
        diction = OrderedDict(sorted(diction.items(), key=lambda x: self.sort_key_raw(x)))
        return diction

    def save_checkpoint(self, save_path: str, tokens: Tensor | list, tag: str, number: int, **kwargs) -> str:
        save_path = os.path.join(save_path, f"{number}.safetensors")
        fake_diction = self.load_checkpoint_for_saving(tag)
        diction = self.tokenizer.detokenize(fake_diction, tokens)
        diction = OrderedDict({k: v.to(self.dtype).contiguous() for k, v in diction.items()})
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=False)
        save_safetensors(diction, save_path)
        return os.path.dirname(os.path.abspath(save_path))

    @staticmethod
    def sort_key(item: tuple) -> float:
        key = item[0]
        number_string = key.split(".")[2]
        if number_string != "weight":
            number = int(number_string)
            if "input_layernorm" in key:
                return number + 0.1
            if "self_attn.q_proj" in key:
                return number + 0.2
            if "self_attn.k_proj" in key:
                return number + 0.3
            if "self_attn.v_proj" in key:
                return number + 0.4
            if "self_attn.o_proj" in key:
                return number + 0.5
            if "post_attention_layernorm" in key:
                return number + 0.6
            if "mlp.up_proj" in key:
                return number + 0.7
            if "mlp.gate_proj" in key:
                return number + 0.8
            if "mlp.down_proj" in key:
                return number + 0.9
            raise RuntimeError(f"Unexpected key {key}")
        else:  # is weight
            if key.split(".")[1] == "embed_tokens":
                return -2
            if key.split(".")[1] == "norm":
                return -1
            raise RuntimeError(f"Unexpected key {key}")

    @staticmethod
    def sort_key_raw(item: tuple) -> float:
        key = item[0]
        number_string = key.split(".")[4]
        if number_string != "weight":
            number = int(number_string)
            if "input_layernorm" in key:
                return number + 0.1
            if "self_attn.q_proj" in key:
                return number + 0.2
            if "self_attn.k_proj" in key:
                return number + 0.3
            if "self_attn.v_proj" in key:
                return number + 0.4
            if "self_attn.o_proj" in key:
                return number + 0.5
            if "post_attention_layernorm" in key:
                return number + 0.6
            if "mlp.up_proj" in key:
                return number + 0.7
            if "mlp.gate_proj" in key:
                return number + 0.8
            if "mlp.down_proj" in key:
                return number + 0.9
            raise RuntimeError(f"Unexpected key {key}")
        else:  # is weight
            if key.split(".")[1] == "embed_tokens":
                return -2
            if key.split(".")[1] == "norm":
                return -1
            raise RuntimeError(f"Unexpected key {key}")


class Text2Qwen25LoRA_MixDataset(Text2Qwen25LoRA_CheckpointDataset):
    dtype = torch.bfloat16

    def __init__(
        self,
        checkpoint_folders: str,
        tokenizer,
        condition_paths: str,
        expected_iteration: int,
        num_texts: int,
        real_length: int = None,
        tokenizer_extra_config: dict = {},
        dataset_extra_config: dict = {},
    ):
        self.length = self.real_length if expected_iteration is None else (expected_iteration + 1000)
        self.tokenizer = tokenizer
        self.condition = [
            torch.load(condition_path, weights_only=True, map_location="cpu") for condition_path in condition_paths
        ]
        self.num_texts = num_texts
        # load checkpoints
        self.checkpoints = [
            os.path.join(f, item)
            for f in checkpoint_folders
            for item in os.listdir(f)[: min(len(os.listdir(f)), real_length)]
        ]

        self.real_length = len(self.checkpoints)
        self.length_for_each = real_length

        self.tokenizer_extra_config = tokenizer_extra_config
        self.dataset_extra_config = dataset_extra_config

    def extract_condition(self, idx: int) -> Tensor:
        # We have extracted conditions externally and we only need to load at this step.

        cls = idx // self.length_for_each

        return self.condition[cls]

    def __getitem__(self, item):
        try:  # If encounter some error, we return the next data.
            checkpoint_path = self.checkpoints[item % self.real_length]
            condition = self.extract_condition(item % self.real_length)
            diction = self.load_checkpoint(checkpoint_path)
            tokens, others = self.tokenizer.tokenize(diction, **self.tokenizer_extra_config)
        except DataLoadingError:
            return self[random.randint(0, self.real_length - 1)]
        return tokens, condition, checkpoint_path

    def __len__(self):
        return self.length


class Text2Qwen25LoRA_FullCondDataset(Text2Qwen25LoRA_MixDataset):
    dtype = torch.bfloat16

    def __init__(
        self,
        checkpoint_folders: str,
        tokenizer,
        num_texts: int,
        texts: list,
        max_text_length: int,
        text_tokenizer: torch.nn.Module,
        expected_iteration: int = None,
        real_length: int = None,
        tokenizer_extra_config: dict = {},
        dataset_extra_config: dict = {},
    ):
        self.length = real_length if expected_iteration is None else (expected_iteration + 1000)
        self.tokenizer = tokenizer

        self.text_tokenizer = text_tokenizer
        try:
            self.texts = [[t["prompt"] for t in dataset] for dataset in texts]
        except:
            self.texts = [[t["conversations"][0]["value"] for t in dataset] for dataset in texts]
        self.num_texts = num_texts
        self.max_text_length = max_text_length

        # load checkpoints
        self.checkpoints = [
            os.path.join(f, item)
            for f in checkpoint_folders
            for item in os.listdir(f)[: min(len(os.listdir(f)), real_length)]
        ]

        self.real_length = len(self.checkpoints)
        self.length_for_each = real_length

        self.tokenizer_extra_config = tokenizer_extra_config
        self.dataset_extra_config = dataset_extra_config

    def extract_condition_with_specified_text(self, text) -> Tensor:
        conds = self.text_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
        )
        input_ids, attention_mask = [
            conds.input_ids,
        ], [
            conds.attention_mask,
        ]

        return torch.stack(input_ids, dim=0), torch.stack(attention_mask, dim=0)

    def extract_condition(self, idx: int) -> Tensor:
        cls = idx // self.length_for_each
        class_text = self.texts[cls]
        text_conds = random.sample(class_text, self.num_texts)

        conds = self.text_tokenizer(
            text_conds,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
        )

        return conds

    def __getitem__(self, item):
        try:  # If encounter some error, we return the next data.
            checkpoint_path = self.checkpoints[item % self.real_length]
            condition = self.extract_condition(item % self.real_length)
            diction = self.load_checkpoint(checkpoint_path)
            tokens, others = self.tokenizer.tokenize(diction, **self.tokenizer_extra_config)
        except DataLoadingError:
            return self[random.randint(0, self.real_length - 1)]
        return tokens, condition, checkpoint_path

    def collate_fn_train(self, batch):
        tokens, conditions, _ = zip(*batch)
        tokens, input_ids, attention_mask = (
            list(tokens),
            [condition.input_ids for condition in conditions],
            [condition.attention_mask for condition in conditions],
        )
        return torch.stack(tokens, dim=0), torch.stack(input_ids, dim=0), torch.stack(attention_mask, dim=0)

    def collate_fn_test(self, batch):
        assert len(batch) == 1
        tokens, conditions, tag = batch[0]
        tokens, input_ids, attention_mask, tag = (
            [
                tokens,
            ],
            [
                conditions.input_ids,
            ],
            [
                conditions.attention_mask,
            ],
            tag,
        )
        return torch.stack(tokens, dim=0), torch.stack(input_ids, dim=0), torch.stack(attention_mask, dim=0), tag


class Text2Qwen25LoRA_VLenCondDataset(Text2Qwen25LoRA_FullCondDataset):
    dtype = torch.bfloat16

    def __init__(
        self,
        checkpoint_folders: str,
        tokenizer,
        num_texts: int,
        texts: list,
        max_text_length: int,
        text_tokenizer: torch.nn.Module,
        number_of_conditions: int,
        expected_iteration: int = None,
        real_length: int = None,
        tokenizer_extra_config: dict = {},
        dataset_extra_config: dict = {},
    ):
        self.length = real_length if expected_iteration is None else (expected_iteration + 1000)
        self.tokenizer = tokenizer

        self.text_tokenizer = text_tokenizer
        self.texts = [[t["prompt"] for t in dataset][:number_of_conditions] for dataset in texts]
        self.num_texts = num_texts
        self.max_text_length = max_text_length

        # load checkpoints
        self.checkpoints = [
            os.path.join(f, item)
            for f in checkpoint_folders
            for item in os.listdir(f)[: min(len(os.listdir(f)), real_length)]
        ]

        self.real_length = len(self.checkpoints)
        self.length_for_each = real_length

        self.tokenizer_extra_config = tokenizer_extra_config
        self.dataset_extra_config = dataset_extra_config


class Text2Qwen25LoRA_CondQ_ADataset(Text2Qwen25LoRA_FullCondDataset):
    dtype = torch.bfloat16

    def __init__(
        self,
        checkpoint_folders: str,
        tokenizer,
        num_texts: int,
        texts: list,
        max_text_length: int,
        text_tokenizer: torch.nn.Module,
        expected_iteration: int = None,
        real_length: int = None,
        tokenizer_extra_config: dict = {},
        dataset_extra_config: dict = {},
    ):
        self.length = real_length if expected_iteration is None else (expected_iteration + 1000)
        self.tokenizer = tokenizer

        self.text_tokenizer = text_tokenizer
        self.texts = [[t["prompt"] + "answer: " + t["response"] for t in dataset] for dataset in texts]
        self.num_texts = num_texts
        self.max_text_length = max_text_length

        # load checkpoints
        self.checkpoints = [
            os.path.join(f, item)
            for f in checkpoint_folders
            for item in os.listdir(f)[: min(len(os.listdir(f)), real_length)]
        ]

        self.real_length = len(self.checkpoints)
        self.length_for_each = real_length

        self.tokenizer_extra_config = tokenizer_extra_config
        self.dataset_extra_config = dataset_extra_config


class Text2Qwen25LoRA_CondADataset(Text2Qwen25LoRA_FullCondDataset):
    dtype = torch.bfloat16

    def __init__(
        self,
        checkpoint_folders: str,
        tokenizer,
        num_texts: int,
        texts: list,
        max_text_length: int,
        text_tokenizer: torch.nn.Module,
        expected_iteration: int = None,
        real_length: int = None,
        tokenizer_extra_config: dict = {},
        dataset_extra_config: dict = {},
    ):
        self.length = real_length if expected_iteration is None else (expected_iteration + 1000)
        self.tokenizer = tokenizer

        self.text_tokenizer = text_tokenizer
        self.texts = [[t["response"] for t in dataset] for dataset in texts]
        self.num_texts = num_texts
        self.max_text_length = max_text_length

        # load checkpoints
        self.checkpoints = [
            os.path.join(f, item)
            for f in checkpoint_folders
            for item in os.listdir(f)[: min(len(os.listdir(f)), real_length)]
        ]

        self.real_length = len(self.checkpoints)
        self.length_for_each = real_length

        self.tokenizer_extra_config = tokenizer_extra_config
        self.dataset_extra_config = dataset_extra_config


class Text2Qwen25LoRA_CondMixDataset(Text2Qwen25LoRA_FullCondDataset):
    dtype = torch.bfloat16

    def __init__(
        self,
        checkpoint_folders: str,
        tokenizer,
        num_texts: int,
        texts: list,
        max_text_length: int,
        text_tokenizer: torch.nn.Module,
        expected_iteration: int = None,
        real_length: int = None,
        tokenizer_extra_config: dict = {},
        dataset_extra_config: dict = {},
    ):
        self.length = real_length if expected_iteration is None else (expected_iteration + 1000)
        self.tokenizer = tokenizer

        self.text_tokenizer = text_tokenizer
        self.texts = []

        for dataset in texts:
            cls = []
            for t in dataset:
                p = random.random()
                condition = t["prompt"] + "answer: " + t["response"] if p > 0.8 else t["prompt"]
                cls.append(condition)
            self.texts.append(cls)

        self.num_texts = num_texts
        self.max_text_length = max_text_length

        # load checkpoints
        self.checkpoints = [
            os.path.join(f, item)
            for f in checkpoint_folders
            for item in os.listdir(f)[: min(len(os.listdir(f)), real_length)]
        ]

        self.real_length = len(self.checkpoints)
        self.length_for_each = real_length

        self.tokenizer_extra_config = tokenizer_extra_config
        self.dataset_extra_config = dataset_extra_config


class Text2Qwen25LoRA_AllCondDataset(Text2Qwen25LoRA_FullCondDataset):
    def __init__(
        self,
        checkpoint_folders: str,
        tokenizer,
        num_texts: int,
        texts: list,
        max_text_length: int,
        text_tokenizer: torch.nn.Module,
        real_length: int = None,
        expected_iteration: int = None,
        tokenizer_extra_config: dict = {},
        dataset_extra_config: dict = {},
    ):
        super().__init__(
            checkpoint_folders=checkpoint_folders,
            tokenizer=tokenizer,
            expected_iteration=expected_iteration,
            real_length=real_length,
            num_texts=num_texts,
            texts=texts,
            max_text_length=max_text_length,
            text_tokenizer=text_tokenizer,
            tokenizer_extra_config=tokenizer_extra_config,
            dataset_extra_config=dataset_extra_config,
        )
        self.text_index = 0

    def extract_condition(self, idx: int) -> Tensor:
        cls = idx // self.length_for_each
        class_text = self.texts[cls]

        if self.text_index + self.num_texts < len(class_text):
            text_conds = class_text[self.text_index : self.text_index + self.num_texts]
            self.text_index += self.num_texts
        else:
            text_conds = (
                class_text[self.text_index :] + class_text[: self.text_index + self.num_texts - len(class_text)]
            )
            self.text_index = self.text_index + self.num_texts - len(class_text)

        conds = self.text_tokenizer(
            text_conds,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
        )
        return conds
