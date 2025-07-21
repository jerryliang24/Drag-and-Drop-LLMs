import torch

from .tokenizer import LoraTokenizer2D_Qwen, Tokenizer2D


@Tokenizer2D.with_scales_wrapper
class Qwen2505LoRA_Tokenizer2D(LoraTokenizer2D_Qwen):
    def __init__(self, token_size: tuple[int, int] = (32, 128)):
        super().__init__(token_size=token_size)

    def selector(self, key: str) -> tuple[str, list, int]:
        # return layer_name, keys, num_tokens
        key = key.rsplit(".", 2)[0]

        if "mlp.down_proj" in key:
            return "lora_mlp2", [f"{key}.lora_A.weight", f"{key}.lora_B.weight"], 45

        if "mlp.gate_proj" in key or "mlp.up_proj" in key:
            return "lora_mlp1", [f"{key}.lora_A.weight", f"{key}.lora_B.weight"], 45

        if "self_attn.k_proj" in key or "self_attn.v_proj" in key:
            return "lora_kv", [f"{key}.lora_A.weight", f"{key}.lora_B.weight"], 8

        if "self_attn.o_proj" in key or "self_attn.q_proj" in key:
            return "lora_qo", [f"{key}.lora_A.weight", f"{key}.lora_B.weight"], 14

    def _tokenize_lora_qo(self, up_weight, down_weight, num_tokens):
        # up_weight: (8, 896); down_weight: (896, 8)
        assert num_tokens == 14
        down_weight = down_weight.transpose(0, 1)
        token_list_up, scale_list_up = self._tokenize_weightchunkin(up_weight, 7)
        token_list_down, scale_list_down = self._tokenize_weightchunkin(down_weight, 7)
        return token_list_up + token_list_down, scale_list_up + scale_list_down

    def _detokenize_lora_qo(self, fake_up_weight, fake_down_weight, tokens, scales):
        # up_weight: (8, 896); down_weight: (896, 8)
        up_weight = self._detokenize_weightchunkin(fake_up_weight, tokens[:7], scales[:7])[0]
        down_weight = self._detokenize_weightchunkin(fake_down_weight.transpose(0, 1), tokens[7:], scales[7:])[0]
        down_weight = down_weight.transpose(0, 1)
        assert up_weight.shape == fake_up_weight.shape and down_weight.shape == fake_down_weight.shape, (
            f"up_weight.shape={up_weight.shape}, fake_up_weight.shape={fake_up_weight.shape}, "
            f"down_weight.shape={down_weight.shape}, fake_down_weight.shape={fake_down_weight.shape}"
        )
        return up_weight, down_weight

    def _tokenize_lora_kv(self, up_weight, down_weight, num_tokens):
        # up_weight: (8, 896); down_weight: (128, 8)
        assert num_tokens == 8
        down_weight = down_weight.transpose(0, 1)
        token_list_up, scale_list_up = self._tokenize_weightchunkin(up_weight, 7)
        token_list_down, scale_list_down = self._tokenize_weightchunkin(down_weight, 1)
        return token_list_up + token_list_down, scale_list_up + scale_list_down

    def _detokenize_lora_kv(self, fake_up_weight, fake_down_weight, tokens, scales):
        # up_weight: (8, 896); down_weight: (128, 8)
        up_weight = self._detokenize_weightchunkin(fake_up_weight, tokens[:7], scales[:7])[0]
        down_weight = self._detokenize_weightchunkin(fake_down_weight.transpose(0, 1), tokens[7:], scales[7:])[0]
        down_weight = down_weight.transpose(0, 1)
        assert up_weight.shape == fake_up_weight.shape and down_weight.shape == fake_down_weight.shape, (
            f"up_weight.shape={up_weight.shape}, fake_up_weight.shape={fake_up_weight.shape}, "
            f"down_weight.shape={down_weight.shape}, fake_down_weight.shape={fake_down_weight.shape}"
        )
        return up_weight, down_weight

    def _tokenize_lora_mlp1(self, up_weight, down_weight, num_tokens):
        # up_weight: (8, 896); down_weight: (4684, 8)
        assert num_tokens == 45
        down_weight = down_weight.transpose(0, 1)
        token_list_up, scale_list_up = self._tokenize_weightchunkin(up_weight, 7)
        token_list_down, scale_list_down = self._tokenize_weightchunkin(down_weight, 38)
        return token_list_up + token_list_down, scale_list_up + scale_list_down

    def _detokenize_lora_mlp1(self, fake_up_weight, fake_down_weight, tokens, scales):
        # up_weight: (8, 896); down_weight: (4684, 8)
        up_weight = self._detokenize_weightchunkin(fake_up_weight, tokens[:7], scales[:7])[0]
        down_weight = self._detokenize_weightchunkin(fake_down_weight.transpose(0, 1), tokens[7:], scales[7:])[0]
        down_weight = down_weight.transpose(0, 1)
        return up_weight, down_weight

    def _tokenize_lora_mlp2(self, up_weight, down_weight, num_tokens):
        # up_weight: (8, 4684); down_weight: (896, 8)
        assert num_tokens == 45
        down_weight = down_weight.transpose(0, 1)
        token_list_up, scale_list_up = self._tokenize_weightchunkin(up_weight, 38)
        token_list_down, scale_list_down = self._tokenize_weightchunkin(down_weight, 7)
        return token_list_up + token_list_down, scale_list_up + scale_list_down

    def _detokenize_lora_mlp2(self, fake_up_weight, fake_down_weight, tokens, scales):
        # up_weight: (8, 4684); down_weight: (896, 8)
        up_weight = self._detokenize_weightchunkin(fake_up_weight, tokens[:38], scales[:38])[0]
        down_weight = self._detokenize_weightchunkin(fake_down_weight.transpose(0, 1), tokens[38:], scales[38:])[0]
        down_weight = down_weight.transpose(0, 1)
        return up_weight, down_weight

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
        assert in_feature % num_tokens == 0 and num_tokens == len(scales), "this this is"
        fake_weights = torch.chunk(fake_weight, dim=1, chunks=num_tokens)
        weights = []
        for fw, tk, sl in zip(fake_weights, tokens, scales):
            weight = self._detokenize_weight(fw, tk[None], sl[None])[0]
            weights.append(weight)
        weight = torch.cat(weights, dim=1)
        assert weight.shape == fake_weight.shape
        return (weight,)


@Tokenizer2D.with_scales_wrapper
class Qwen2515LoRA_Tokenizer2D(LoraTokenizer2D_Qwen):
    def selector(self, key: str) -> tuple[str, list, int]:
        # return layer_name, keys, num_tokens
        key = key.rsplit(".", 2)[0]

        if "mlp.down_proj" in key:
            return "lora_mlp2", [f"{key}.lora_A.weight", f"{key}.lora_B.weight"], 41

        if "mlp.gate_proj" in key or "mlp.up_proj" in key:
            return "lora_mlp1", [f"{key}.lora_A.weight", f"{key}.lora_B.weight"], 41

        if "self_attn.k_proj" in key or "self_attn.v_proj" in key:
            return "lora_kv", [f"{key}.lora_A.weight", f"{key}.lora_B.weight"], 7

        if "self_attn.o_proj" in key or "self_attn.q_proj" in key:
            return "lora_qo", [f"{key}.lora_A.weight", f"{key}.lora_B.weight"], 12

    def _tokenize_lora_qo(self, up_weight, down_weight, num_tokens):
        # up_weight: (16, 1536); down_weight: (1536, 16)
        assert num_tokens == 12
        down_weight = down_weight.transpose(0, 1)
        token_list_up, scale_list_up = self._tokenize_weightchunkin(up_weight, 6)
        token_list_down, scale_list_down = self._tokenize_weightchunkin(down_weight, 6)
        return token_list_up + token_list_down, scale_list_up + scale_list_down

    def _detokenize_lora_qo(self, fake_up_weight, fake_down_weight, tokens, scales):
        # up_weight: (16, 1536); down_weight: (1536, 16)
        up_weight = self._detokenize_weightchunkin(fake_up_weight, tokens[:6], scales[:6])[0]
        down_weight = self._detokenize_weightchunkin(fake_down_weight.transpose(0, 1), tokens[6:], scales[6:])[0]
        down_weight = down_weight.transpose(0, 1)
        assert up_weight.shape == fake_up_weight.shape and down_weight.shape == fake_down_weight.shape, (
            f"up_weight.shape={up_weight.shape}, fake_up_weight.shape={fake_up_weight.shape}, "
            f"down_weight.shape={down_weight.shape}, fake_down_weight.shape={fake_down_weight.shape}"
        )
        return up_weight, down_weight

    def _tokenize_lora_kv(self, up_weight, down_weight, num_tokens):
        # up_weight: (16, 1536); down_weight: (256, 16)
        assert num_tokens == 7
        down_weight = down_weight.transpose(0, 1)
        token_list_up, scale_list_up = self._tokenize_weightchunkin(up_weight, 6)
        token_list_down, scale_list_down = self._tokenize_weightchunkin(down_weight, 1)
        return token_list_up + token_list_down, scale_list_up + scale_list_down

    def _detokenize_lora_kv(self, fake_up_weight, fake_down_weight, tokens, scales):
        # up_weight: (16, 1536); down_weight: (256, 16)
        up_weight = self._detokenize_weightchunkin(fake_up_weight, tokens[:6], scales[:6])[0]
        down_weight = self._detokenize_weightchunkin(fake_down_weight.transpose(0, 1), tokens[6:], scales[6:])[0]
        down_weight = down_weight.transpose(0, 1)
        assert up_weight.shape == fake_up_weight.shape and down_weight.shape == fake_down_weight.shape, (
            f"up_weight.shape={up_weight.shape}, fake_up_weight.shape={fake_up_weight.shape}, "
            f"down_weight.shape={down_weight.shape}, fake_down_weight.shape={fake_down_weight.shape}"
        )
        return up_weight, down_weight

    def _tokenize_lora_mlp1(self, up_weight, down_weight, num_tokens):
        # up_weight: (16, 1536); down_weight: (8960, 16)
        assert num_tokens == 41
        down_weight = down_weight.transpose(0, 1)
        token_list_up, scale_list_up = self._tokenize_weightchunkin(up_weight, 6)
        token_list_down, scale_list_down = self._tokenize_weightchunkin(down_weight, 35)
        return token_list_up + token_list_down, scale_list_up + scale_list_down

    def _detokenize_lora_mlp1(self, fake_up_weight, fake_down_weight, tokens, scales):
        # up_weight: (16, 1536); down_weight: (8960, 16)
        up_weight = self._detokenize_weightchunkin(fake_up_weight, tokens[:6], scales[:6])[0]
        down_weight = self._detokenize_weightchunkin(fake_down_weight.transpose(0, 1), tokens[6:], scales[6:])[0]
        down_weight = down_weight.transpose(0, 1)
        return up_weight, down_weight

    def _tokenize_lora_mlp2(self, up_weight, down_weight, num_tokens):
        # up_weight: (16, 8960); down_weight: (1536, 16)
        assert num_tokens == 41
        down_weight = down_weight.transpose(0, 1)
        token_list_up, scale_list_up = self._tokenize_weightchunkin(up_weight, 35)
        token_list_down, scale_list_down = self._tokenize_weightchunkin(down_weight, 6)
        return token_list_up + token_list_down, scale_list_up + scale_list_down

    def _detokenize_lora_mlp2(self, fake_up_weight, fake_down_weight, tokens, scales):
        # up_weight: (16, 8960); down_weight: (1536, 16)
        up_weight = self._detokenize_weightchunkin(fake_up_weight, tokens[:35], scales[:35])[0]
        down_weight = self._detokenize_weightchunkin(fake_down_weight.transpose(0, 1), tokens[35:], scales[35:])[0]
        down_weight = down_weight.transpose(0, 1)
        return up_weight, down_weight

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
        assert in_feature % num_tokens == 0 and num_tokens == len(scales), "this this is"
        fake_weights = torch.chunk(fake_weight, dim=1, chunks=num_tokens)
        weights = []
        for fw, tk, sl in zip(fake_weights, tokens, scales):
            weight = self._detokenize_weight(fw, tk[None], sl[None])[0]
            weights.append(weight)
        weight = torch.cat(weights, dim=1)
        assert weight.shape == fake_weight.shape
        return (weight,)


@Tokenizer2D.with_scales_wrapper
class Qwen253BVL_LoRA_Tokenizer2D(LoraTokenizer2D_Qwen):
    def selector(self, key: str) -> tuple[str, list, int]:
        # return layer_name, keys, num_tokens
        key = key.rsplit(".", 2)[0]

        if "mlp.down_proj" in key:
            return "lora_mlp2", [f"{key}.lora_A.weight", f"{key}.lora_B.weight"], 51

        if "mlp.gate_proj" in key or "mlp.up_proj" in key:
            return "lora_mlp1", [f"{key}.lora_A.weight", f"{key}.lora_B.weight"], 51

        if "self_attn.k_proj" in key or "self_attn.v_proj" in key:
            return "lora_kv", [f"{key}.lora_A.weight", f"{key}.lora_B.weight"], 9

        if "self_attn.o_proj" in key or "self_attn.q_proj" in key:
            return "lora_qo", [f"{key}.lora_A.weight", f"{key}.lora_B.weight"], 16

    def _tokenize_lora_qo(self, up_weight, down_weight, num_tokens):
        # up_weight: (16, 2048); down_weight: (2048, 16)
        assert num_tokens == 16
        down_weight = down_weight.transpose(0, 1)
        token_list_up, scale_list_up = self._tokenize_weightchunkin(up_weight, 8)
        token_list_down, scale_list_down = self._tokenize_weightchunkin(down_weight, 8)
        return token_list_up + token_list_down, scale_list_up + scale_list_down

    def _detokenize_lora_qo(self, fake_up_weight, fake_down_weight, tokens, scales):
        up_weight = self._detokenize_weightchunkin(fake_up_weight, tokens[:8], scales[:8])[0]
        down_weight = self._detokenize_weightchunkin(fake_down_weight.transpose(0, 1), tokens[8:], scales[8:])[0]
        down_weight = down_weight.transpose(0, 1)
        assert up_weight.shape == fake_up_weight.shape and down_weight.shape == fake_down_weight.shape, (
            f"up_weight.shape={up_weight.shape}, fake_up_weight.shape={fake_up_weight.shape}, "
            f"down_weight.shape={down_weight.shape}, fake_down_weight.shape={fake_down_weight.shape}"
        )
        return up_weight, down_weight

    def _tokenize_lora_kv(self, up_weight, down_weight, num_tokens):
        # up_weight: (16, 2048); down_weight: (256, 16)
        assert num_tokens == 9
        down_weight = down_weight.transpose(0, 1)
        token_list_up, scale_list_up = self._tokenize_weightchunkin(up_weight, 8)
        token_list_down, scale_list_down = self._tokenize_weightchunkin(down_weight, 1)
        return token_list_up + token_list_down, scale_list_up + scale_list_down

    def _detokenize_lora_kv(self, fake_up_weight, fake_down_weight, tokens, scales):
        up_weight = self._detokenize_weightchunkin(fake_up_weight, tokens[:8], scales[:8])[0]
        down_weight = self._detokenize_weightchunkin(fake_down_weight.transpose(0, 1), tokens[8:], scales[8:])[0]
        down_weight = down_weight.transpose(0, 1)
        assert up_weight.shape == fake_up_weight.shape and down_weight.shape == fake_down_weight.shape, (
            f"up_weight.shape={up_weight.shape}, fake_up_weight.shape={fake_up_weight.shape}, "
            f"down_weight.shape={down_weight.shape}, fake_down_weight.shape={fake_down_weight.shape}"
        )
        return up_weight, down_weight

    def _tokenize_lora_mlp1(self, up_weight, down_weight, num_tokens):
        # up_weight: (16, 2048); down_weight: (11008, 16)
        assert num_tokens == 51
        down_weight = down_weight.transpose(0, 1)
        token_list_up, scale_list_up = self._tokenize_weightchunkin(up_weight, 8)
        token_list_down, scale_list_down = self._tokenize_weightchunkin(down_weight, 43)
        return token_list_up + token_list_down, scale_list_up + scale_list_down

    def _detokenize_lora_mlp1(self, fake_up_weight, fake_down_weight, tokens, scales):
        up_weight = self._detokenize_weightchunkin(fake_up_weight, tokens[:8], scales[:8])[0]
        down_weight = self._detokenize_weightchunkin(fake_down_weight.transpose(0, 1), tokens[8:], scales[8:])[0]
        down_weight = down_weight.transpose(0, 1)
        return up_weight, down_weight

    def _tokenize_lora_mlp2(self, up_weight, down_weight, num_tokens):
        # up_weight: (16, 11008); down_weight: (2048, 16)
        assert num_tokens == 51
        down_weight = down_weight.transpose(0, 1)
        token_list_up, scale_list_up = self._tokenize_weightchunkin(up_weight, 43)
        token_list_down, scale_list_down = self._tokenize_weightchunkin(down_weight, 8)
        return token_list_up + token_list_down, scale_list_up + scale_list_down

    def _detokenize_lora_mlp2(self, fake_up_weight, fake_down_weight, tokens, scales):
        up_weight = self._detokenize_weightchunkin(fake_up_weight, tokens[:43], scales[:43])[0]
        down_weight = self._detokenize_weightchunkin(fake_down_weight.transpose(0, 1), tokens[43:], scales[43:])[0]
        down_weight = down_weight.transpose(0, 1)
        return up_weight, down_weight

    def _tokenize_weightchunkin(self, weight, num_tokens):
        out_feature, in_feature = weight.shape
        self.token_size[1]
        if in_feature % num_tokens == 0:
            weights = torch.chunk(weight, dim=1, chunks=num_tokens)
        else:
            raise ValueError
        tokens_list, scales_list = [], []
        for w in weights:
            tokens, scales = self._tokenize_weight(weight=w, num_tokens=1)
            tokens_list.extend(tokens)
            scales_list.extend(scales)
        return tokens_list, scales_list

    def _detokenize_weightchunkin(self, fake_weight, tokens, scales):
        out_feature, in_feature = fake_weight.shape
        num_tokens = len(tokens)
        assert in_feature % num_tokens == 0 and num_tokens == len(scales), "this this is"
        fake_weights = torch.chunk(fake_weight, dim=1, chunks=num_tokens)
        weights = []
        for fw, tk, sl in zip(fake_weights, tokens, scales):
            weight = self._detokenize_weight(fw, tk[None], sl[None])[0]
            weights.append(weight)
        weight = torch.cat(weights, dim=1)
        assert weight.shape == fake_weight.shape
        return (weight,)


@Tokenizer2D.with_scales_wrapper
class Qwen257BLoRA_Tokenizer2D(LoraTokenizer2D_Qwen):
    def selector(self, key: str) -> tuple[str, list, int]:
        # return layer_name, keys, num_tokens
        key = key.rsplit(".", 2)[0]

        if "mlp.down_proj" in key:
            return "lora_mlp2", [f"{key}.lora_A.weight", f"{key}.lora_B.weight"], 44

        if "mlp.gate_proj" in key or "mlp.up_proj" in key:
            return "lora_mlp1", [f"{key}.lora_A.weight", f"{key}.lora_B.weight"], 44

        if "self_attn.k_proj" in key or "self_attn.v_proj" in key:
            return "lora_kv", [f"{key}.lora_A.weight", f"{key}.lora_B.weight"], 8

        if "self_attn.o_proj" in key or "self_attn.q_proj" in key:
            return "lora_qo", [f"{key}.lora_A.weight", f"{key}.lora_B.weight"], 14

    def _tokenize_lora_qo(self, up_weight, down_weight, num_tokens):
        # up_weight: (16, 3584); down_weight: (3584, 16)
        assert num_tokens == 14
        down_weight = down_weight.transpose(0, 1)
        token_list_up, scale_list_up = self._tokenize_weightchunkin(up_weight, 7)
        token_list_down, scale_list_down = self._tokenize_weightchunkin(down_weight, 7)
        return token_list_up + token_list_down, scale_list_up + scale_list_down

    def _detokenize_lora_qo(self, fake_up_weight, fake_down_weight, tokens, scales):
        # up_weight: (8, 896); down_weight: (896, 8)
        up_weight = self._detokenize_weightchunkin(fake_up_weight, tokens[:7], scales[:7])[0]
        down_weight = self._detokenize_weightchunkin(fake_down_weight.transpose(0, 1), tokens[7:], scales[7:])[0]
        down_weight = down_weight.transpose(0, 1)
        assert up_weight.shape == fake_up_weight.shape and down_weight.shape == fake_down_weight.shape, (
            f"up_weight.shape={up_weight.shape}, fake_up_weight.shape={fake_up_weight.shape}, "
            f"down_weight.shape={down_weight.shape}, fake_down_weight.shape={fake_down_weight.shape}"
        )
        return up_weight, down_weight

    def _tokenize_lora_kv(self, up_weight, down_weight, num_tokens):
        # up_weight: (8, 896); down_weight: (128, 8)
        assert num_tokens == 8
        down_weight = down_weight.transpose(0, 1)
        token_list_up, scale_list_up = self._tokenize_weightchunkin(up_weight, 7)
        token_list_down, scale_list_down = self._tokenize_weightchunkin(down_weight, 1)
        return token_list_up + token_list_down, scale_list_up + scale_list_down

    def _detokenize_lora_kv(self, fake_up_weight, fake_down_weight, tokens, scales):
        # up_weight: (8, 896); down_weight: (128, 8)
        up_weight = self._detokenize_weightchunkin(fake_up_weight, tokens[:7], scales[:7])[0]
        down_weight = self._detokenize_weightchunkin(fake_down_weight.transpose(0, 1), tokens[7:], scales[7:])[0]
        down_weight = down_weight.transpose(0, 1)
        assert up_weight.shape == fake_up_weight.shape and down_weight.shape == fake_down_weight.shape, (
            f"up_weight.shape={up_weight.shape}, fake_up_weight.shape={fake_up_weight.shape}, "
            f"down_weight.shape={down_weight.shape}, fake_down_weight.shape={fake_down_weight.shape}"
        )
        return up_weight, down_weight

    def _tokenize_lora_mlp1(self, up_weight, down_weight, num_tokens):
        # up_weight: (8, 896); down_weight: (4684, 8)
        assert num_tokens == 44
        down_weight = down_weight.transpose(0, 1)
        token_list_up, scale_list_up = self._tokenize_weightchunkin(up_weight, 7)
        token_list_down, scale_list_down = self._tokenize_weightchunkin(down_weight, 37)
        return token_list_up + token_list_down, scale_list_up + scale_list_down

    def _detokenize_lora_mlp1(self, fake_up_weight, fake_down_weight, tokens, scales):
        # up_weight: (16, 3584); down_weight: (18944, 16)
        up_weight = self._detokenize_weightchunkin(fake_up_weight, tokens[:7], scales[:7])[0]
        down_weight = self._detokenize_weightchunkin(fake_down_weight.transpose(0, 1), tokens[7:], scales[7:])[0]
        down_weight = down_weight.transpose(0, 1)
        return up_weight, down_weight

    def _tokenize_lora_mlp2(self, up_weight, down_weight, num_tokens):
        # up_weight: (16, 18944); down_weight: (3584, 16)
        assert num_tokens == 44
        down_weight = down_weight.transpose(0, 1)
        token_list_up, scale_list_up = self._tokenize_weightchunkin(up_weight, 37)
        token_list_down, scale_list_down = self._tokenize_weightchunkin(down_weight, 7)
        return token_list_up + token_list_down, scale_list_up + scale_list_down

    def _detokenize_lora_mlp2(self, fake_up_weight, fake_down_weight, tokens, scales):
        # up_weight: (8, 4684); down_weight: (896, 8)
        up_weight = self._detokenize_weightchunkin(fake_up_weight, tokens[:37], scales[:37])[0]
        down_weight = self._detokenize_weightchunkin(fake_down_weight.transpose(0, 1), tokens[37:], scales[37:])[0]
        down_weight = down_weight.transpose(0, 1)
        return up_weight, down_weight

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
        assert in_feature % num_tokens == 0 and num_tokens == len(scales), "this this is"
        fake_weights = torch.chunk(fake_weight, dim=1, chunks=num_tokens)
        weights = []
        for fw, tk, sl in zip(fake_weights, tokens, scales):
            weight = self._detokenize_weight(fw, tk[None], sl[None])[0]
            weights.append(weight)
        weight = torch.cat(weights, dim=1)
        assert weight.shape == fake_weight.shape
        return (weight,)
