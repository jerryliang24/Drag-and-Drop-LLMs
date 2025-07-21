from typing import Optional

import torch
from torch import nn


class Text_Encoder(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_texts: int,
        seq_len: int,
        emb_dim: int,
    ):
        super(Text_Encoder, self).__init__()
        self.backbone = backbone
        self.num_texts = num_texts
        self.seq_len = seq_len
        self.proj = nn.Linear(emb_dim, emb_dim * seq_len)
        nn.init.normal_(self.proj.weight, mean=0.0, std=1.0)

    def forward(self, text):
        embedding = self.backbone.encode(text)

        C = embedding.shape[-1]
        embedding = torch.tensor(embedding).view(-1, self.num_texts, C)
        out = self.proj(embedding).view(-1, self.num_texts, self.seq_len, C)

        return out


class LLM_Encoder(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        tokenizer,
        num_texts: int,
        seq_len: int,
    ):
        super(Text_Encoder, self).__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.num_texts = num_texts
        self.seq_len = seq_len

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

    def forward(self, text):
        raw_conds = self.text_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.seq_len,
        )

        input_ids = raw_conds.input_ids
        attention_mask = raw_conds.attention_mask

        outputs = self.feature_extraction_with_specified_layer(
            layer_index=self.layer_index,
            input_ids=input_ids.view(-1, self.seq_len),
            attention_mask=attention_mask.view(-1, self.seq_len),
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=False,
            output_attentions=False,
            cache_position=None,
            **{},
        )
        return outputs.view(B, N, L, -1)
