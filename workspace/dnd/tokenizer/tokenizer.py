import types
from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import deepcopy
from functools import partial, reduce
from operator import mul
from typing import Any

import torch
from torch import Tensor

cumprod = partial(reduce, mul)


class TokenizerInterface(ABC):
    @abstractmethod
    def tokenize(self, diction: OrderedDict, **kwargs) -> tuple[Tensor, Any]:
        """
        Tokenize checkpoints to tokens or feature maps.
        :param diction: The ordered diction loaded from torch.load(...).
        :param kwargs: These parameters come from dataset(tokenizer_extra_config:dict).
        :return: (tokens: Tensor, others, any other things (if no, output an extra None))
        """

    @abstractmethod
    def detokenize(self, fake_diction: OrderedDict, tokens: Tensor, **kwargs) -> OrderedDict:
        """
        Detokenize checkpoints from tokens or feature maps.
        :param fake_diction: A fake diction to provide structure info.
        :param tokens: tokens, the same as tokens tokenized by self.tokenize.
        :param kwargs: These parameters come from dataset(tokenizer_extra_config:dict).
        :return: OrderedDict, the same as inputted diction.
        """

    @staticmethod
    def pad_to_shape_2d(tensor, shape, padding_value=torch.nan):
        assert tensor.dim() == 2, "pad_to_shape_2d only receive tensor in 2D."
        output_tensor = torch.full(size=shape, fill_value=padding_value)
        output_tensor[0 : tensor.size(0), 0 : tensor.size(1)] = tensor
        return output_tensor


class Tokenizer2D(TokenizerInterface):
    padding_value = torch.nan

    def __init__(self, token_size: [tuple[int, int], int]):
        self.token_size = (token_size, token_size) if isinstance(token_size, int) else token_size

    def tokenize(self, diction: OrderedDict, **kwargs) -> tuple[Tensor, Any]:
        tokens_list, scales_list = [], []
        while diction:
            layer_name, keys, num_tokens = self.selector(next(iter(diction)), diction)
            function = getattr(self, f"_tokenize_{layer_name}")
            tokens, scales = function(*[diction.pop(key) for key in keys], num_tokens)
            assert len(tokens) == num_tokens == len(scales) // 4, (
                f"Token number should be {num_tokens} according to selector, "
                f"but got len(tokens)={len(tokens)}, len(scales)={len(scales)}, layer_name={layer_name}"
            )
            tokens_list.extend(tokens)
            scales_list.extend(scales)
        tokens = torch.stack(tokens_list, dim=0)
        scales = torch.tensor(scales_list).view(tokens.size(0), 4)
        return tokens, scales

    def detokenize(self, fake_diction: OrderedDict, tokens: Tensor, scales: Tensor = None, **kwargs) -> OrderedDict:
        diction = OrderedDict()
        while fake_diction:
            layer_name, keys, num_tokens = self.selector(next(iter(fake_diction)), fake_diction)
            function = getattr(self, f"_detokenize_{layer_name}")
            params = function(
                *[fake_diction.pop(key) for key in keys], tokens=tokens[:num_tokens], scales=scales[:num_tokens]
            )  # detokenize function
            tokens, scales = tokens[num_tokens:], scales[num_tokens:]
            assert len(params) == len(keys), (
                f"Params number should be {len(keys)} according to selector, \n"
                f"but got len(params)={len(params)}, len(keys)={len(keys)}, keys={keys}"
            )
            diction.update({k: v for k, v in zip(keys, params)})
        return diction

    def selector(self, key: str, diction: OrderedDict) -> tuple[str, list, int]:
        # return layer_name, keys, num_tokens
        raise NotImplementedError("You need to override selector function")

    def _tokenize_norm(self, weight, bias, num_tokens=1):
        assert weight.dim() == 1 and bias.dim() == 1 and num_tokens == 1
        weight_mean, weight_std = weight.mean(), weight.std()
        bias_mean, bias_std = bias.mean(), bias.std()
        weight, bias = (weight - weight_mean) / weight_std, (bias - bias_mean) / bias_std
        weight_matrix = torch.full_like(torch.diag_embed(weight), self.padding_value)
        weight_matrix.diagonal(dim1=-2, dim2=-1).copy_(weight)
        token = torch.cat((weight_matrix, torch.unsqueeze(bias, dim=-1)), dim=-1)
        token = self.pad_to_shape_2d(token, self.token_size)
        return [
            token,
        ], [weight_mean, weight_std, bias_mean, bias_std]

    def _detokenize_norm(self, fake_weight, fake_bias, tokens, scales):
        assert fake_weight.dim() == 1 and fake_bias.dim() == 1 and len(tokens) == 1 and len(scales) == 1
        tokens, scales = torch.squeeze(tokens, dim=0), torch.squeeze(scales, dim=0)
        weight_mean, weight_std, bias_mean, bias_std = scales
        tokens = tokens[0 : fake_weight.numel(), 0 : fake_weight.numel() + 1]
        weight, bias = tokens[:, :-1], tokens[:, -1]
        weight = weight.diagonal(dim1=-2, dim2=-1)
        weight, bias = weight * weight_std + weight_mean, bias * bias_std + bias_mean
        assert weight.shape == fake_weight.shape, bias.shape == fake_bias.shape
        return weight, bias

    def _tokenize_batchnorm(self, weight, bias, num_batches, num_tokens):
        weight = (torch.pow(weight, 1.0 / 3.0) - 5.0) / 8.0
        bias = torch.pow(bias.abs(), 1.0 / 3.0) * (bias / bias.abs())
        return self._tokenize_norm(weight, bias, num_tokens)

    def _detokenize_batchnorm(self, fake_weight, fake_bias, num_batches, tokens, scales):
        weight, bias = self._detokenize_norm(fake_weight, fake_bias, tokens, scales)
        weight = torch.pow(torch.clip(weight * 8.0 + 5.0, min=0.3), 3.0)
        bias = torch.pow(bias, 3.0)
        return weight, bias, num_batches

    def _tokenize_normweight(self, weight, num_tokens=1):
        assert weight.dim() == 1 and num_tokens == 1
        weight_mean, weight_std = weight.mean(), weight.std()
        weight = (weight - weight_mean) / weight_std
        weight_matrix = torch.full_like(torch.diag_embed(weight), self.padding_value)
        weight_matrix.diagonal(dim1=-2, dim2=-1).copy_(weight)
        token = self.pad_to_shape_2d(weight_matrix, self.token_size)
        return [
            token,
        ], [weight_mean, weight_std, weight_mean, weight_std]

    def _detokenize_normweight(self, fake_weight, tokens, scales):
        assert fake_weight.dim() == 1 and len(tokens) == 1 and len(scales) == 1
        tokens, scales = torch.squeeze(tokens, dim=0), torch.squeeze(scales, dim=0)
        weight_mean, weight_std = (scales[0] + scales[2]) / 2, (scales[1] + scales[3]) / 2
        tokens = tokens[0 : fake_weight.numel(), 0 : fake_weight.numel() + 1]
        weight = tokens[:, :-1].diagonal(dim1=-2, dim2=-1)
        weight = weight * weight_std + weight_mean
        assert weight.shape == fake_weight.shape
        return (weight,)

    def _tokenize_linear(self, weight, bias, num_tokens=1):
        assert weight.dim() == 2 and bias.dim() == 1 and num_tokens == 1
        weight_mean, weight_std = weight.mean(), weight.std()
        weight_std = 1.0 if torch.isnan(weight_std) else weight_std
        bias_mean, bias_std = bias.mean(), bias.std()
        bias_std = 1.0 if torch.isnan(bias_std) else bias_std
        weight, bias = (weight - weight_mean) / weight_std, (bias - bias_mean) / bias_std
        token = torch.cat((weight, torch.unsqueeze(bias, dim=-1)), dim=-1)
        token = self.pad_to_shape_2d(token, self.token_size, padding_value=self.padding_value)
        return [
            token,
        ], [weight_mean, weight_std, bias_mean, bias_std]

    def _detokenize_linear(self, fake_weight, fake_bias, tokens, scales):
        assert fake_weight.dim() == 2 and fake_bias.dim() == 1 and len(tokens) == 1 and len(scales) == 1
        tokens, scales = torch.squeeze(tokens, dim=0), torch.squeeze(scales, dim=0)
        weight_mean, weight_std, bias_mean, bias_std = scales
        tokens = tokens[0 : fake_weight.size(0), 0 : fake_weight.size(1) + 1]
        weight, bias = tokens[:, :-1], tokens[:, -1]
        weight, bias = weight * weight_std + weight_mean, bias * bias_std + bias_mean
        assert weight.shape == fake_weight.shape, bias.shape == fake_bias.shape
        return weight, bias

    def _tokenize_weight(self, weight, num_tokens=1):
        assert weight.dim() == 2 and num_tokens == 1
        weight_mean, weight_std = weight.mean(), weight.std()
        weight_std = 1.0 if torch.isnan(weight_std) else weight_std
        weight = (weight - weight_mean) / weight_std
        token = self.pad_to_shape_2d(weight, self.token_size, padding_value=self.padding_value)
        return [
            token,
        ], [weight_mean, weight_std, weight_mean, weight_std]

    def _detokenize_weight(self, fake_weight, tokens, scales):
        assert fake_weight.dim() == 2 and len(tokens) == 1 and len(scales) == 1
        tokens, scales = torch.squeeze(tokens, dim=0), torch.squeeze(scales, dim=0)
        weight_mean, weight_std = (
            (scales[0] + scales[2]) / 2,
            (scales[1] + scales[3]) / 2,
        )
        weight = tokens[0 : fake_weight.size(0), 0 : fake_weight.size(1)]
        weight = weight * weight_std + weight_mean
        assert weight.shape == fake_weight.shape
        return (weight,)

    def _tokenize_conv(self, weight, bias, num_tokens):
        weight_permute = torch.permute(weight.flatten(start_dim=-2), (2, 0, 1))
        assert len(weight_permute) == num_tokens
        tokens, scales = [], []
        for weight in weight_permute:
            token, scale = self._tokenize_linear(weight, bias, num_tokens=1)
            tokens.extend(token)
            scales.extend(scale)
        return tokens, scales

    def _detokenize_conv(self, fake_weight, fake_bias, tokens, scales):
        fake_weight_permute = torch.permute(fake_weight.flatten(start_dim=-2), (2, 0, 1))
        weights, biases = [], []
        for fw, tk, sl in zip(fake_weight_permute, tokens, scales):
            weight, bias = self._detokenize_linear(fw, fake_bias, tk[None], sl[None])
            weights.append(weight)
            biases.append(bias)
        weight = torch.stack(weights, dim=-1).unflatten(dim=-1, sizes=fake_weight.shape[-2:])
        bias = torch.stack(biases, dim=0).mean(dim=0)
        assert weight.shape == fake_weight.shape and bias.shape == fake_bias.shape
        return weight, bias

    def _tokenize_convweight(self, weight, num_tokens):
        weight_permute = torch.permute(weight.flatten(start_dim=-2), (2, 0, 1))
        assert len(weight_permute) == num_tokens
        tokens, scales = [], []
        for weight in weight_permute:
            token, scale = self._tokenize_weight(weight, num_tokens=1)
            tokens.extend(token)
            scales.extend(scale)
        return tokens, scales

    def _detokenize_convweight(self, fake_weight, tokens, scales):
        fake_weight_permute = torch.permute(fake_weight.flatten(start_dim=-2), (2, 0, 1))
        weights = []
        for fw, tk, sl in zip(fake_weight_permute, tokens, scales):
            weight = self._detokenize_weight(fw, tk[None], sl[None])[0]
            weights.append(weight)
        weight = torch.stack(weights, dim=-1).unflatten(dim=-1, sizes=fake_weight.shape[-2:])
        assert weight.shape == fake_weight.shape
        return (weight,)

    def _tokenize_chunkout(self, weight, bias, num_tokens):
        out_feature, in_feature = weight.shape
        assert out_feature % num_tokens == 0
        weights = torch.chunk(weight, dim=0, chunks=num_tokens)
        biases = torch.chunk(bias, dim=0, chunks=num_tokens)
        tokens_list, scales_list = [], []
        for w, b in zip(weights, biases):
            tokens, scales = self._tokenize_linear(weight=w, bias=b, num_tokens=1)
            tokens_list.extend(tokens)
            scales_list.extend(scales)
        return tokens_list, scales_list

    def _detokenize_chunkout(self, fake_weight, fake_bias, tokens, scales):
        out_feature, in_feature = fake_weight.shape
        num_tokens = len(tokens)
        assert out_feature % num_tokens == 0 and num_tokens == len(scales)
        fake_weights = torch.chunk(fake_weight, dim=0, chunks=num_tokens)
        fake_biases = torch.chunk(fake_bias, dim=0, chunks=num_tokens)
        weights, biases = [], []
        for fw, fb, tk, sl in zip(fake_weights, fake_biases, tokens, scales):
            weight, bias = self._detokenize_linear(fw, fb, tk[None], sl[None])
            weights.append(weight)
            biases.append(bias)
        weight = torch.cat(weights, dim=0)
        bias = torch.cat(biases, dim=0)
        assert (
            weight.shape == fake_weight.shape and bias.shape == fake_bias.shape
        ), f"weight={weight.shape}, bias={bias.shape}, fake_weight={fake_weight.shape}, fake_bias={fake_bias.shape}"
        return weight, bias

    def _tokenize_chunkin(self, weight, bias, num_tokens):
        out_feature, in_feature = weight.shape
        assert in_feature % num_tokens == 0
        weights = torch.chunk(weight, dim=1, chunks=num_tokens)
        tokens_list, scales_list = [], []
        for w in weights:
            tokens, scales = self._tokenize_linear(weight=w, bias=bias, num_tokens=1)
            tokens_list.extend(tokens)
            scales_list.extend(scales)
        return tokens_list, scales_list

    def _detokenize_chunkin(self, fake_weight, fake_bias, tokens, scales):
        out_feature, in_feature = fake_weight.shape
        num_tokens = len(tokens)
        assert in_feature % num_tokens == 0 and num_tokens == len(scales)
        fake_weights = torch.chunk(fake_weight, dim=1, chunks=num_tokens)
        weights, biases = [], []
        for fw, tk, sl in zip(fake_weights, tokens, scales):
            weight, bias = self._detokenize_linear(fw, fake_bias, tk[None], sl[None])
            weights.append(weight)
            biases.append(bias)
        weight = torch.cat(weights, dim=1)
        bias = torch.stack(biases, dim=0).mean(dim=0)
        assert (
            weight.shape == fake_weight.shape and bias.shape == fake_bias.shape
        ), f"weight={weight.shape}, bias={bias.shape}, fake_weight={fake_weight.shape}, fake_bias={fake_bias.shape}"
        return weight, bias

    def _tokenize_weightchunkout(self, weight, num_tokens):
        out_feature, in_feature = weight.shape
        assert out_feature % num_tokens == 0
        weights = torch.chunk(weight, dim=0, chunks=num_tokens)
        tokens_list, scales_list = [], []
        for w in weights:
            tokens, scales = self._tokenize_weight(weight=w, num_tokens=1)
            tokens_list.extend(tokens)
            scales_list.extend(scales)
        return tokens_list, scales_list

    def _detokenize_weightchunkout(self, fake_weight, tokens, scales):
        out_feature, in_feature = fake_weight.shape
        num_tokens = len(tokens)
        assert out_feature % num_tokens == 0 and num_tokens == len(scales)
        fake_weights = torch.chunk(fake_weight, dim=0, chunks=num_tokens)
        weights, biases = [], []
        for fw, tk, sl in zip(fake_weights, tokens, scales):
            weight = self._detokenize_weight(fw, tk[None], sl[None])[0]
            weights.append(weight)
        weight = torch.cat(weights, dim=0)
        assert weight.shape == fake_weight.shape
        return (weight,)

    def _tokenize_weightchunkin(self, weight, num_tokens):
        out_feature, in_feature = weight.shape
        assert in_feature % num_tokens == 0
        weights = torch.chunk(weight, dim=1, chunks=num_tokens)
        tokens_list, scales_list = [], []
        for w in weights:
            tokens, scales = self._tokenize_weight(weight=w, num_tokens=1)
            tokens_list.extend(tokens)
            scales_list.extend(scales)
        return tokens_list, scales_list

    def _detokenize_weightchunkin(self, fake_weight, tokens, scales):
        out_feature, in_feature = fake_weight.shape
        num_tokens = len(tokens)
        assert in_feature % num_tokens == 0 and num_tokens == len(scales)
        fake_weights = torch.chunk(fake_weight, dim=1, chunks=num_tokens)
        weights = []
        for fw, tk, sl in zip(fake_weights, tokens, scales):
            weight = self._detokenize_weight(fw, tk[None], sl[None])[0]
            weights.append(weight)
        weight = torch.cat(weights, dim=1)
        assert weight.shape == fake_weight.shape
        return (weight,)

    @staticmethod
    def with_scales_wrapper(cls):
        def wrapped_class(
            token_size: [tuple[int, int], int], mean_scale: float = 8.0, std_bias: float = 1.6, *args, **kwargs
        ):
            obj = cls(token_size, *args, **kwargs)
            ori_tokenize = deepcopy(obj.tokenize)
            ori_detokenize = deepcopy(obj.detokenize)

            def tokenize(self, diction: OrderedDict, **kwargs) -> tuple[Tensor, Any]:
                tokens, scales = ori_tokenize(diction=diction, **kwargs)
                scales = scales.view(tokens.size(0), 2, 2)
                scales[:, :, 0] = scales[:, :, 0] * mean_scale
                scales[:, :, 1] = torch.log(scales[:, :, 1] * 0.9 + 0.1) + std_bias
                tokens[:, -2:, :] = scales.repeat((1, 1, tokens.size(-1) // 2))
                tokens[:, :, -2:] = scales.repeat((1, tokens.size(-2) // 2, 1))
                return tokens, None

            def detokenize(self, fake_diction: OrderedDict, tokens: Tensor, **kwargs) -> OrderedDict:
                scales_width = tokens[:, -2:, :].unfold(dimension=-1, size=2, step=2).mean(dim=2)
                scales_height = tokens[:, :, -2:].reshape(tokens.size(0), -1, 2, 2).mean(dim=1)
                scales = (scales_width + scales_height) * 0.5
                scales[:, :, 0] = scales[:, :, 0] / mean_scale
                scales[:, :, 1] = (torch.exp(scales[:, :, 1] - std_bias) - 0.1) / 0.9
                scales = scales.view(scales.size(0), 4)
                return ori_detokenize(fake_diction=fake_diction, tokens=tokens, scales=scales, **kwargs)

            # reattach
            obj.tokenize = types.MethodType(tokenize, obj)
            obj.detokenize = types.MethodType(detokenize, obj)
            return obj

        return wrapped_class


class LoraTokenizer2D(TokenizerInterface):
    padding_value = torch.nan

    def __init__(self, token_size: [tuple[int, int], int], *, _use_scales=False):
        self.token_size = (token_size, token_size) if isinstance(token_size, int) else token_size

    def tokenize(self, diction: OrderedDict, **kwargs) -> tuple[Tensor, Any]:
        tokens_list, scales_list = [], []
        index = 0
        while diction:
            layer_name, keys, num_tokens = self.selector(next(iter(diction)), diction)
            function = getattr(self, f"_tokenize_{layer_name}")
            tokens, scales = function(*[diction.pop(key) for key in keys], num_tokens)
            assert len(tokens) == num_tokens == len(scales), (
                f"Token number should be {num_tokens} according to selector, "
                f"but got len(tokens)={len(tokens)}, len(scales)={len(scales)}, layer_name={layer_name}"
            )
            tokens_list.extend(tokens)
            scales_list.extend(scales)
            index += num_tokens
        tokens = torch.stack(tokens_list, dim=0)
        tokens = tokens.view(len(tokens_list) // 2, 2, tokens.size(1), tokens.size(2))
        tokens.transpose_(0, 1)
        return tokens, None

    def detokenize(self, fake_diction: OrderedDict, tokens: Tensor, scales: Tensor = None, **kwargs) -> OrderedDict:
        assert tokens.size(0) == 2
        up_tokens, down_tokens = tokens[0], tokens[1]
        tokens = torch.stack((up_tokens, down_tokens), dim=1).flatten(end_dim=1)
        scales = torch.ones_like(up_tokens[:, -1:, :].mean(dim=(-1, -2)).view(-1, 1).repeat(1, 2).flatten(end_dim=1))
        # Note: we do not use scales here
        diction = OrderedDict()
        while fake_diction:
            layer_name, keys, num_tokens = self.selector(next(iter(fake_diction)), fake_diction)
            function = getattr(self, f"_detokenize_{layer_name}")
            params = function(
                *[fake_diction.pop(key) for key in keys], tokens=tokens[:num_tokens], scales=scales[:num_tokens]
            )  # detokenize function
            tokens, scales = tokens[num_tokens:], scales[num_tokens:]
            assert len(params) == len(keys), (
                f"Params number should be {len(keys)} according to selector, \n"
                f"but got len(params)={len(params)}, len(keys)={len(keys)}, keys={keys}"
            )
            diction.update({k: v for k, v in zip(keys, params)})
        return diction

    def _tokenize_weight(self, weight, num_tokens=1, std=None):
        assert weight.dim() == 2 and num_tokens == 1
        weight_std = torch.full((1,), fill_value=self.padding_value)
        # weight = weight / weight_std  # Note: we do not use scales here
        token = self.pad_to_shape_2d(weight, self.token_size, padding_value=self.padding_value)
        return [
            token,
        ], [
            weight_std,
        ]

    def _detokenize_weight(self, fake_weight, tokens, scales):
        assert fake_weight.dim() == 2 and len(tokens) == 1 and scales.flatten().numel() == 1
        tokens, weight_std = torch.squeeze(tokens, dim=0), scales.flatten()
        weight = tokens[0 : fake_weight.size(0), 0 : fake_weight.size(1)]
        weight = weight * weight_std
        assert weight.shape == fake_weight.shape, f"weight.shape={weight.shape}, fake_weight.shape={fake_weight.shape}"
        return (weight,)

    def _tokenize_lora(self, up_weight, down_weight, num_tokens):
        # up_weight: (1280, 32); down_weight: (32, 1280)
        assert num_tokens == 2
        up_weight = up_weight.transpose(0, 1)
        token_list_up, scale_list_up = self._tokenize_weight(up_weight, 1)
        token_list_down, scale_list_down = self._tokenize_weight(down_weight, 1)
        return token_list_up + token_list_down, scale_list_up + scale_list_down

    def _detokenize_lora(self, fake_up_weight, fake_down_weight, tokens, scales):
        # up_weight: (1280, 32); down_weight: (32, 1280)
        assert len(tokens) == 2 and len(scales) == 2
        up_weight = self._detokenize_weight(fake_up_weight.transpose(0, 1), tokens[:1], scales[:1])[0]
        down_weight = self._detokenize_weight(fake_down_weight, tokens[1:], scales[1:])[0]
        up_weight = up_weight.transpose(0, 1)
        assert up_weight.shape == fake_up_weight.shape and down_weight.shape == fake_down_weight.shape, (
            f"up_weight.shape={up_weight.shape}, fake_up_weight.shape={fake_up_weight.shape}, "
            f"down_weight.shape={down_weight.shape}, fake_down_weight.shape={fake_down_weight.shape}"
        )
        return up_weight, down_weight


class LoraTokenizer2D_Qwen(TokenizerInterface):
    padding_value = torch.nan

    def __init__(self, token_size: [tuple[int, int], int], *, _use_scales=False):
        self.token_size = (token_size, token_size) if isinstance(token_size, int) else token_size

    def tokenize(self, diction: OrderedDict, **kwargs) -> tuple[Tensor, Any]:
        tokens_list = []
        scales_list = []
        for k, v in diction.items():
            # we process lora_A and lora_B altogether
            if "lora_B" in k:
                continue

            layer_name, keys, num_tokens = self.selector(k)
            function = getattr(self, f"_tokenize_{layer_name}")
            LoRAs = [diction[key] for key in keys]
            tokens, scales = function(*LoRAs, num_tokens)
            assert len(tokens) == num_tokens, (
                f"Token number should be {num_tokens} according to selector, "
                f"but got len(tokens) = {len(tokens)}, layer_name={layer_name}"
            )
            tokens_list.extend(tokens)
            scales_list.extend(scales)
        tokens = torch.stack(tokens_list, dim=0)
        scales = torch.tensor(scales_list).view(tokens.size(0), 4)
        return tokens, scales

    def detokenize(self, fake_diction: OrderedDict, tokens: Tensor, scales: Tensor = None, **kwargs) -> OrderedDict:
        diction = {}
        for k, v in fake_diction.items():
            # we process lora_A and lora_B altogether
            if "lora_B" in k:
                continue

            layer_name, keys, num_tokens = self.selector(k)
            function = getattr(self, f"_detokenize_{layer_name}")
            params = function(
                *[fake_diction[key] for key in keys], tokens=tokens[:num_tokens], scales=scales[:num_tokens]
            )  # detokenize function
            tokens, scales = tokens[num_tokens:], scales[num_tokens:]
            assert len(params) == len(keys), (
                f"Params number should be {len(keys)} according to selector, \n"
                f"but got len(params)={len(params)}, len(keys)={len(keys)}, keys={keys}"
            )
            diction.update({k: v for k, v in zip(keys, params)})
        return diction

    def _tokenize_weight(self, weight, num_tokens=1):
        assert weight.dim() == 2 and num_tokens == 1
        weight_mean, weight_std = weight.mean(), weight.std()
        weight_std = 1.0 if torch.isnan(weight_std) else weight_std
        weight = (weight - weight_mean) / weight_std
        token = self.pad_to_shape_2d(weight, self.token_size, padding_value=self.padding_value)
        return [
            token,
        ], [weight_mean, weight_std, weight_mean, weight_std]

    def _detokenize_weight(self, fake_weight, tokens, scales):
        assert fake_weight.dim() == 2 and len(tokens) == 1 and len(scales) == 1
        tokens, scales = torch.squeeze(tokens, dim=0), torch.squeeze(scales, dim=0)
        weight_mean, weight_std = (
            (scales[0] + scales[2]) / 2,
            (scales[1] + scales[3]) / 2,
        )
        weight = tokens[0 : fake_weight.size(0), 0 : fake_weight.size(1)]
        weight = weight * weight_std + weight_mean
        assert weight.shape == fake_weight.shape
        return (weight,)
