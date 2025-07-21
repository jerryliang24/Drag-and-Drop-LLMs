import os
import random
from abc import ABC, abstractmethod
from collections import OrderedDict

import torch
from torch import Tensor
from torch.utils.data import Dataset

from workspace.dnd.tokenizer import TokenizerInterface
from workspace.dnd.tools import load_safetensors


class DataLoadingError(Exception):
    pass


class DatasetInterface(ABC, Dataset):
    @abstractmethod
    def __getitem__(self, item) -> tuple[Tensor, Tensor, str]:
        """
        Get item from dataset
        :param item: int
        :return: token, condition, tag: Tensor, Tensor, str
        """

    @abstractmethod
    def __len__(self) -> int:
        """
        Get length of dataset
        :return: length (maybe a fake length), int
        """

    @abstractmethod
    def load_checkpoint(self, load_path: str) -> OrderedDict:
        """
        Load checkpoint from path
        :param load_path: file or folder to load the checkpoint
        :return: diction got from state_dict()
        """

    @abstractmethod
    def save_checkpoint(self, save_path: str, tokens: Tensor | list, tag: str, **kwargs) -> None:
        """
        Save checkpoint from tokens
        :param save_path: file or folder to save the checkpoint
        :param tokens: generated tokens
        :param tag: a tag for finding other information
        :param kwargs: interface for future dataset class
        """

    @abstractmethod
    def extract_condition(self, checkpoint_path: str) -> Tensor | None:
        """
        Extract condition from checkpoint path
        :param checkpoint_path: the path to one checkpoint
        """


class CheckpointDataset(DatasetInterface):
    def __init__(
        self,
        checkpoint_folder: str,
        tokenizer: TokenizerInterface,
        *,
        expected_iteration: int = None,
        tokenizer_extra_config: dict = {},
        dataset_extra_config: dict = {},
        real_length: int = None,
    ):
        self.checkpoints = []
        for checkpoint_name in os.listdir(checkpoint_folder):
            checkpoint_path = os.path.join(checkpoint_folder, checkpoint_name)
            self.checkpoints.append(checkpoint_path)
        self.tokenizer = tokenizer
        self.real_length = len(self.checkpoints) if real_length is None else real_length
        self.length = self.real_length if expected_iteration is None else (expected_iteration + 1000)
        self.tokenizer_extra_config = tokenizer_extra_config
        self.dataset_extra_config = dataset_extra_config
        # init for some kinds of tokenizer
        self.training = True

    def __getitem__(self, item):
        try:  # If encounter some error, we return the next data.
            checkpoint_path = self.checkpoints[item % self.real_length]
            condition = self.extract_condition(checkpoint_path=checkpoint_path)
            diction = self.load_checkpoint(checkpoint_path)
            tokens, others = self.tokenizer.tokenize(diction, **self.tokenizer_extra_config)
        except DataLoadingError:
            return self[random.randint(0, self.real_length - 1)]
        return tokens, condition, checkpoint_path

    def __len__(self):
        return self.length

    def load_checkpoint(self, load_path: str) -> OrderedDict:
        return torch.load(load_path, map_location="cpu", weights_only=True)

    def save_checkpoint(self, save_path: str, tokens: Tensor | list, tag: str, **kwargs) -> None:
        fake_diction = self.load_checkpoint(tag)
        diction = self.tokenizer.detokenize(fake_diction, tokens)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=False)
        torch.save(diction, save_path)
        return os.path.abspath(save_path)

    @staticmethod
    def collate_fn_train(batch):
        tokens, conditions, tag = zip(*batch)
        return list(tokens), list(conditions)

    @staticmethod
    def collate_fn_test(batch):
        assert len(batch) == 1
        tokens, conditions, tag = batch[0]
        return (
            [
                tokens,
            ],
            [
                conditions,
            ],
            tag,
        )


class CheckpointDataset_PaddedToSame(CheckpointDataset):
    @staticmethod
    def collate_fn_train(batch):
        tokens, conditions = CheckpointDataset.collate_fn_train(batch)
        return torch.stack(tokens, dim=0), torch.stack(conditions, dim=0)

    @staticmethod
    def collate_fn_test(batch):
        tokens, conditions, tag = CheckpointDataset.collate_fn_test(batch)
        return torch.stack(tokens, dim=0), torch.stack(conditions, dim=0), tag

    def extract_condition(self, checkpoint_path) -> Tensor | None:
        """
        Extract condition from checkpoint path
        :return: None, as this dataset does not use condition
        """
        return None

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
