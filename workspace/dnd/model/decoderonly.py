from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from workspace.dnd.module.hyperconv import HyperConvDecoder

from .interface import ModelInterface

BERT_MAX_L = 512


class HyperConvDecoderModel(ModelInterface):
    config = {
        "features": [(1024, 16, 16), (320, 320, 175), (288, 288, 175)],
        "condition_dim": (2048, 16, 16),
        "kernel_size": 5,
    }

    def __init__(
        self,
        config,
        criterion_weight: Tensor,
        extra_condition_module: nn.Module = nn.Identity(),
        *,
        freeze_extra_condition=False,
        return_individual_loss=False,
    ):
        super().__init__(config)
        self.return_individual_loss = return_individual_loss
        self.extra_condition = extra_condition_module
        if freeze_extra_condition:
            for param in self.extra_condition.parameters():
                param.requires_grad = False
        self.model = HyperConvDecoder(**self.config)
        self.register_buffer("criterion_weight", criterion_weight)

    def forward(
        self, source: Tensor = None, mask: Tensor = None, condition: Tensor = None, target: Tensor = None, **kwargs
    ) -> Tensor:
        if kwargs.get("generate") is not None:
            return self.generate(source, mask, condition, target, **kwargs)
        condition = self.extra_condition(condition)
        if kwargs.get("noise_enhance") is not None:
            condition += torch.randn_like(condition) * kwargs.get("noise_enhance")
            target += torch.randn_like(target) * kwargs.get("noise_enhance") * 0.1
        output = self.model(condition)
        loss = self.criterion(output, target, mask)
        return loss

    @torch.inference_mode()
    def generate(
        self, source: Tensor = None, mask: Tensor = None, condition: Tensor = None, target: Tensor = None, **kwargs
    ) -> Tensor:
        condition = self.extra_condition(condition)
        output = self.model(condition)
        return output

    def criterion(self, output, target, mask):
        output[~mask], target[~mask] = 0.0, 0.0
        loss = F.mse_loss(output, target, reduction="none") * self.criterion_weight
        if self.return_individual_loss:
            loss = [torch.mean(l[m]) for l, m in zip(loss, mask)]
            return loss
        else:  # normal mean loss
            loss = torch.mean(loss[mask])
            return loss


class HyperConvDecoderModel_FullCond(HyperConvDecoderModel):
    def __init__(
        self,
        config,
        criterion_weight: Tensor,
        extractor_type: str,
        extra_condition_module: nn.Module = nn.Identity(),
        *,
        freeze_extra_condition=True,
        return_individual_loss=False,
        layer_index=None,
    ):
        super().__init__(config, criterion_weight)
        self.return_individual_loss = return_individual_loss

        self.condition_module = extra_condition_module
        if freeze_extra_condition:
            for param in self.condition_module.parameters():
                param.requires_grad = False
        self.model = HyperConvDecoder(**self.config)
        self.extractor_type = extractor_type
        self.layer_index = layer_index

    def extract_condition(self, raw_condition):
        if self.extractor_type == "LLM":
            input_ids = raw_condition["input_ids"]
            attention_mask = raw_condition["attention_mask"]

            B, N, L = input_ids.shape

            outputs = self.feature_extraction_with_specified_layer(
                layer_index=self.layer_index,
                input_ids=input_ids.view(-1, L),
                attention_mask=attention_mask.view(-1, L),
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                use_cache=False,
                output_attentions=False,
                cache_position=None,
                **{},
            )
            return outputs.view(B, N, L, -1)

        elif self.extractor_type == "BERT":
            input_ids = raw_condition["input_ids"]
            attention_mask = raw_condition["attention_mask"]

            B, N, L = input_ids.shape
            input_ids = input_ids.view(-1, L)
            attention_mask = attention_mask.view(-1, L)

            if L > BERT_MAX_L:
                M = L // BERT_MAX_L
                L = BERT_MAX_L
                input_ids = input_ids.view(-1, M, L).view(-1, L)
                attention_mask = attention_mask.view(-1, M, L).view(-1, L)
            else:
                M = 1

            conditions = self.condition_module(input_ids=input_ids, attention_mask=attention_mask)

            return conditions.last_hidden_state.view(B, N, L * M, -1)

        elif self.extractor_type == "Glove":
            return raw_condition["input_ids"]

        elif self.extractor_type == "T5":
            input_ids = raw_condition["input_ids"]
            attention_mask = raw_condition["attention_mask"]

            B, N, L = input_ids.shape

            input_ids = input_ids.view(-1, L)
            attention_mask = attention_mask.view(-1, L)

            conditions = self.condition_module(input_ids=input_ids, attention_mask=attention_mask)
            print(conditions.last_hidden_state.view(B, N, L, -1).shape)
            return conditions.last_hidden_state.view(B, N, L, -1)

        else:
            raise NotImplementedError

    def feature_extraction_with_specified_layer(
        self,
        layer_index: int = -1,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs,
    ):
        # this refer to Mistral Model's forward function
        model = self.condition_module.to(self.device)
        output_attentions = output_attentions if output_attentions is not None else model.config.output_attentions
        use_cache = use_cache if use_cache is not None else model.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)

        if inputs_embeds is None:
            inputs_embeds = model.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = model._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds
        # create position embeddings to be shared across the decoder layers
        position_embeddings = model.rotary_emb(hidden_states, position_ids)

        # decoder layers
        with torch.no_grad():
            for layer_id, decoder_layer in enumerate(model.layers[: layer_index + 1]):
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

                hidden_states = layer_outputs[0]

        return hidden_states

    def forward(
        self, source: Tensor = None, mask: Tensor = None, condition: Tensor = None, target: Tensor = None, **kwargs
    ) -> Tensor:
        if kwargs.get("generate") is not None:
            return self.generate(source, mask, condition, target, **kwargs)
        condition = self.extract_condition(condition)
        if kwargs.get("noise_enhance") is not None:
            condition += torch.randn_like(condition) * kwargs.get("noise_enhance")
            target += torch.randn_like(target) * kwargs.get("noise_enhance") * 0.1
        output = self.model(condition)
        loss = self.criterion(output, target, mask)
        return loss

    @torch.inference_mode()
    def generate(
        self, source: Tensor = None, mask: Tensor = None, condition: Tensor = None, target: Tensor = None, **kwargs
    ) -> Tensor:
        condition = self.extract_condition(condition)
        output = self.model(condition)
        return output


class HyperConvDecoderModel_SuperLarge(HyperConvDecoderModel_FullCond):
    def __init__(
        self,
        config,
        criterion_weight: Tensor,
        extractor_type: str,
        max_length: int,
        modified_length: int,
        extra_condition_module: nn.Module = nn.Identity(),
        *,
        freeze_extra_condition=True,
        return_individual_loss=False,
    ):
        super().__init__(
            config=config,
            criterion_weight=criterion_weight,
            extractor_type=extractor_type,
            extra_condition_module=extra_condition_module,
            freeze_extra_condition=freeze_extra_condition,
            return_individual_loss=return_individual_loss,
        )
        self.down_proj = nn.Linear(max_length, modified_length)
        self.gate_proj = nn.Linear(modified_length, modified_length)
        nn.init.normal_(self.down_proj.weight, 0, 1)
        nn.init.normal_(self.gate_proj.weight, 0, 1)

    def forward(
        self, source: Tensor = None, mask: Tensor = None, condition: Tensor = None, target: Tensor = None, **kwargs
    ) -> Tensor:
        if kwargs.get("generate") is not None:
            return self.generate(source, mask, condition, target, **kwargs)
        condition = self.extract_condition(condition)

        B, N, L, C = condition.shape
        condition = self.gate_proj(self.down_proj(condition.view(B, N, C, L))).view(B, N, -1, C)

        if kwargs.get("noise_enhance") is not None:
            condition += torch.randn_like(condition) * kwargs.get("noise_enhance")
            target += torch.randn_like(target) * kwargs.get("noise_enhance") * 0.1
        output = self.model(condition)

        loss = self.criterion(output, target, mask)
        return loss

    @torch.inference_mode()
    def generate(
        self, source: Tensor = None, mask: Tensor = None, condition: Tensor = None, target: Tensor = None, **kwargs
    ) -> Tensor:
        condition = self.extract_condition(condition)
        B, N, L, C = condition.shape
        condition = self.gate_proj(self.down_proj(condition.view(B, N, C, L))).view(B, N, -1, C)
        output = self.model(condition)
        return output
