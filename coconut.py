# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json
from collections import namedtuple
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.models.gpt2 import GPT2LMHeadModel

Outputs = namedtuple(
    "Outputs",
    ["loss", "inputs_embeds", "logits", "align_loss", "kge_embedding", "anchor_coverage"],
)
MAX_N_LATENT = 8


class Coconut(nn.Module):
    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
        pad_token_id,
        kge_config: Optional[dict] = None,
    ):
        super(Coconut, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = -100
        self.use_kge = False
        self.align_loss_weight = 0.0
        self.latent_injection = "none"

        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()

        if kge_config and kge_config.get("use_kge", False):
            self._init_kge_modules(kge_config)

    def _load_kge_embeddings(self, kge_config: dict) -> torch.Tensor:
        artifact_root = Path(kge_config["kge_artifact_root"])
        embedding_file = kge_config.get("kge_entity_embeddings_file", "entity_embeddings.pt")
        metadata_file = kge_config.get("kge_metadata_file", "metadata.json")

        embedding_path = artifact_root / embedding_file
        if not embedding_path.exists():
            raise FileNotFoundError(f"Missing KGE entity embedding file: {embedding_path}")
        entity_embeddings = torch.load(embedding_path, map_location="cpu")
        if not isinstance(entity_embeddings, torch.Tensor):
            raise TypeError(
                f"Expected entity embeddings at {embedding_path} to be a torch.Tensor"
            )

        if torch.is_complex(entity_embeddings):
            entity_embeddings = torch.view_as_real(entity_embeddings).reshape(
                entity_embeddings.shape[0], -1
            )

        entity_embeddings = entity_embeddings.float()

        metadata_path = artifact_root / metadata_file
        if metadata_path.exists():
            with metadata_path.open() as f:
                metadata = json.load(f)
            expected_dim = metadata.get("projector_in_dim")
            if expected_dim is not None and int(expected_dim) != entity_embeddings.shape[1]:
                raise ValueError(
                    "KGE embedding dim mismatch with metadata projector_in_dim: "
                    f"{entity_embeddings.shape[1]} vs {expected_dim}"
                )

        return entity_embeddings

    def _init_kge_modules(self, kge_config: dict):
        hidden_size = self.embedding.embedding_dim
        entity_embeddings = self._load_kge_embeddings(kge_config)
        kge_input_dim = entity_embeddings.shape[1]

        projector_hidden = kge_config.get("kge_projector_hidden", hidden_size)
        num_hidden_layers = int(kge_config.get("kge_projector_num_hidden_layers", 1))
        activation = kge_config.get("kge_projector_activation", "gelu").lower()
        use_layernorm = bool(kge_config.get("kge_projector_layernorm", True))

        if num_hidden_layers < 1:
            raise ValueError(
                f"kge_projector_num_hidden_layers must be >= 1, got {num_hidden_layers}"
            )

        if activation not in {"gelu", "relu", "none"}:
            raise ValueError(f"Unsupported kge_projector_activation='{activation}'")

        def _append_activation():
            if activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "relu":
                layers.append(nn.ReLU())

        layers = [nn.Linear(kge_input_dim, projector_hidden)]
        _append_activation()

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(projector_hidden, projector_hidden))
            _append_activation()

        layers.append(nn.Linear(projector_hidden, hidden_size))
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_size))

        self.kge_projector = nn.Sequential(*layers)
        self.kge_residual_norm = nn.LayerNorm(hidden_size)

        self.register_buffer("kge_entity_embeddings", entity_embeddings)
        self.use_kge = True
        self.align_loss_weight = kge_config.get("align_loss_weight", 0.0)
        self.latent_injection = kge_config.get("latent_injection", "residual")

    def project_entity_ids(self, entity_ids: torch.Tensor) -> torch.Tensor:
        if not self.use_kge:
            raise RuntimeError("KGE projection requested but use_kge=False")

        max_id = self.kge_entity_embeddings.shape[0] - 1
        safe_ids = entity_ids.clamp(min=0, max=max_id)
        embeddings = self.kge_entity_embeddings[safe_ids]

        if embeddings.ndim == 2:
            return self.kge_projector(embeddings)
        if embeddings.ndim == 3:
            batch, anchors, dim = embeddings.shape
            projected = self.kge_projector(embeddings.reshape(-1, dim))
            return projected.view(batch, anchors, -1)
        raise ValueError(f"Unsupported entity_ids lookup shape {tuple(embeddings.shape)}")

    def _apply_anchor_substitution(
        self,
        inputs_embeds: torch.Tensor,
        anchor_entity_ids: Optional[torch.Tensor],
        anchor_token_spans: Optional[torch.Tensor],
        anchor_mask: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.use_kge:
            return None, None, None, None

        if anchor_entity_ids is None or anchor_token_spans is None or anchor_mask is None:
            raise ValueError(
                "KGE conditioning requested but anchor_entity_ids/anchor_token_spans/anchor_mask are missing."
            )

        num_entities = self.kge_entity_embeddings.shape[0]
        valid_lookup = anchor_mask & (anchor_entity_ids >= 0) & (anchor_entity_ids < num_entities)
        anchor_coverage = valid_lookup.float().mean()

        if not valid_lookup.any():
            return None, None, valid_lookup.any(dim=1), anchor_coverage

        projected = self.project_entity_ids(anchor_entity_ids)
        batch_size, _, hidden_size = projected.shape

        seq_len = inputs_embeds.shape[1]
        span_valid = valid_lookup.clone()

        for batch_idx in range(batch_size):
            for anchor_idx in range(projected.shape[1]):
                if not bool(valid_lookup[batch_idx, anchor_idx]):
                    continue
                start, end = anchor_token_spans[batch_idx, anchor_idx].tolist()
                if start < 0 or end <= start or end > seq_len:
                    span_valid[batch_idx, anchor_idx] = False
                    continue
                replacement = projected[batch_idx, anchor_idx].view(1, hidden_size)
                inputs_embeds[batch_idx, start:end, :] = replacement

        valid_for_pool = span_valid.unsqueeze(-1).to(dtype=projected.dtype)
        pooled = (projected * valid_for_pool).sum(dim=1)
        denom = valid_for_pool.sum(dim=1).clamp(min=1.0)
        pooled = pooled / denom
        pooled = pooled.to(dtype=self.kge_residual_norm.weight.dtype)
        pooled = self.kge_residual_norm(pooled)
        has_anchor = span_valid.any(dim=1)
        return pooled, projected, has_anchor, anchor_coverage

    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        position_ids,
        anchor_entity_ids=None,
        anchor_token_spans=None,
        anchor_mask=None,
        analysis_trace: Optional[dict] = None,
        analysis_batch_index: int = 0,
        **kwargs,
    ):
        analysis_enabled = analysis_trace is not None

        batch_size = input_ids.shape[0]
        inputs_embeds = self.embedding(input_ids)

        kge_embedding, _, has_anchor, anchor_coverage = self._apply_anchor_substitution(
            inputs_embeds=inputs_embeds,
            anchor_entity_ids=anchor_entity_ids,
            anchor_token_spans=anchor_token_spans,
            anchor_mask=anchor_mask,
        )

        if analysis_enabled:
            if analysis_batch_index < 0 or analysis_batch_index >= batch_size:
                raise IndexError(
                    f"analysis_batch_index={analysis_batch_index} is out of range for batch size {batch_size}"
                )
            analysis_trace.setdefault("trajectory", [])

        logits = []

        latent_indices = (input_ids == self.latent_token_id).nonzero()

        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]

        max_n_latents = max([len(l) for l in latent_lists])

        next_compute_range = (0, input_ids.shape[1])

        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())

        kv_cache = None

        for pass_idx in range(max_n_latents):
            if kv_cache is None:
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    output_hidden_states=True,
                )
                hidden_states_offset = 0

            else:
                past_key_values = [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]

                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[:, : next_compute_range[1]],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                )

                hidden_states_offset = next_compute_range[0]

            logits.append(outputs.logits)

            next_compute_range = (
                next_compute_range[1],
                (
                    input_ids.shape[1]
                    if pass_idx + 1 >= max_n_latents
                    else next_compute_range[1] + 1
                ),
            )

            hidden_states = outputs.hidden_states[-1]
            kv_cache = outputs.past_key_values

            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]

            tensor_list = [
                [
                    inputs_embeds[batch_idx, pos, :]
                    for pos in range(inputs_embeds.shape[1])
                ]
                for batch_idx in range(inputs_embeds.shape[0])
            ]

            for batch_idx, token_idx in filling_indices:
                new_state = hidden_states[
                    batch_idx, token_idx - 1 - hidden_states_offset, :
                ]
                if (
                    kge_embedding is not None
                    and self.latent_injection == "residual"
                    and pass_idx == 0
                    and bool(has_anchor[batch_idx])
                ):
                    new_state = new_state + kge_embedding[batch_idx]

                if analysis_enabled and batch_idx == analysis_batch_index:
                    analysis_trace.setdefault("trajectory", []).append(
                        new_state.detach().cpu()
                    )
                tensor_list[batch_idx][token_idx] = new_state

            inputs_embeds = torch.stack(
                [
                    torch.stack(tensor_list[batch_idx])
                    for batch_idx in range(inputs_embeds.shape[0])
                ]
            )

        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[
                :, next_compute_range[0] : next_compute_range[1], :
            ],
            attention_mask=attention_mask[:, : next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
            past_key_values=(
                [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]
                if kv_cache
                else None
            ),
            output_hidden_states=True,
        )

        logits.append(outputs.logits)

        if kge_embedding is not None and self.align_loss_weight > 0:
            pre_decode_hidden = outputs.hidden_states[-1][:, -1, :]
            valid = has_anchor if has_anchor is not None else torch.zeros(
                (pre_decode_hidden.shape[0],), dtype=torch.bool, device=pre_decode_hidden.device
            )
            if valid.any():
                norm_proj = F.normalize(kge_embedding[valid], dim=-1)
                norm_hidden = F.normalize(pre_decode_hidden[valid], dim=-1)
                align_loss = 1 - F.cosine_similarity(norm_proj, norm_hidden, dim=-1)
                align_loss = align_loss.mean()
            else:
                align_loss = None
        else:
            align_loss = None

        self.gen_forward_cnt += max_n_latents + 1

        logits = torch.cat(logits, dim=-2)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        if align_loss is not None:
            loss = loss + self.align_loss_weight * align_loss

        return Outputs(
            loss=loss,
            inputs_embeds=inputs_embeds,
            logits=logits,
            align_loss=align_loss,
            kge_embedding=kge_embedding,
            anchor_coverage=anchor_coverage,
        )

    def train(self):
        self.base_causallm.train()

    def eval(self):
        self.base_causallm.eval()

    def generate(
        self,
        input_ids,
        attention_mask,
        max_new_tokens=16,
        output_embedding=False,
        synced_gpus=False,
        anchor_entity_ids=None,
        anchor_token_spans=None,
        anchor_mask=None,
        analysis_trace: Optional[dict] = None,
        analysis_batch_index: int = 0,
        **kwargs,
    ):
        self.gen_forward_cnt = 0

        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"

        tokens = input_ids[0].detach().tolist()

        labels = input_ids.clone()
        outputs = self.forward(
            input_ids,
            torch.ones_like(input_ids, device=input_ids.device),
            labels,
            torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).reshape(1, -1),
            anchor_entity_ids=anchor_entity_ids,
            anchor_token_spans=anchor_token_spans,
            anchor_mask=anchor_mask,
            analysis_trace=analysis_trace,
            analysis_batch_index=analysis_batch_index,
        )
        inputs_embeds = outputs.inputs_embeds

        next_token = torch.argmax(outputs.logits[0, -1]).item()
        tokens.append(next_token)
        new_token_embed = self.embedding(
            torch.tensor(next_token, device=input_ids.device)
        ).view(1, 1, -1)
        new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)

        for _ in range(max_new_tokens - 1):
            outputs = self.base_causallm(inputs_embeds=new_inputs_embeds)
            self.gen_forward_cnt += 1
            next_token = torch.argmax(outputs.logits[0, -1]).item()
            if next_token == self.eos_token_id:
                break
            tokens.append(next_token)
            new_token_embed = self.embedding(
                torch.tensor(next_token, device=input_ids.device)
            ).view(1, 1, -1)
            new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)

        if synced_gpus:
            while self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT:
                self.gen_forward_cnt += 1
                _ = self.base_causallm(inputs_embeds=new_inputs_embeds)

        if output_embedding:
            return torch.tensor(tokens).view(1, -1), new_inputs_embeds

        return torch.tensor(tokens).view(1, -1)
