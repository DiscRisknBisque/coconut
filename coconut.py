# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from typing import Optional

from transformers.models.gpt2 import GPT2LMHeadModel

from graph_encoder import GraphPrefixAdapter, GraphProjector, build_graph_encoder

Outputs = namedtuple(
    "Outputs",
    ["loss", "inputs_embeds", "logits", "align_loss", "graph_embedding"],
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
        graph_config: Optional[dict] = None,
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
        self.use_graph = False
        self.graph_prefix_len = 0
        self.align_loss_weight = 0.0
        self.latent_injection = "none"
        self.graph_dim = None

        # tested with GPT2 and Llama3
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()

        if graph_config and graph_config.get("use_graph", False):
            self._init_graph_modules(graph_config)

    def _init_graph_modules(self, graph_config: dict):
        hidden_size = self.embedding.embedding_dim
        graph_dim = graph_config.get("graph_dim", 256)
        graph_input_dim = graph_config.get(
            "graph_input_dim", graph_config.get("graph_feature_dim", 259)
        )
        graph_hidden_dim = graph_config.get("graph_hidden_dim", graph_dim)
        graph_encoder_name = graph_config.get("graph_encoder", "gcn2")
        prefix_len = graph_config.get("graph_prefix_len", 4)
        projector_hidden = graph_config.get("graph_projector_hidden", graph_dim)
        dropout = graph_config.get("graph_dropout", 0.1)
        projector_dropout = graph_config.get("graph_projector_dropout", dropout)
        prefix_dropout = graph_config.get("graph_prefix_dropout", dropout)

        self.graph_encoder = build_graph_encoder(
            graph_encoder_name,
            d_in=graph_input_dim,
            d_hidden=graph_hidden_dim,
            d_out=graph_dim,
            dropout=dropout,
        )
        self.graph_projector = GraphProjector(
            d_in=graph_dim,
            d_hidden=projector_hidden,
            d_out=hidden_size,
            dropout=projector_dropout,
        )
        self.graph_prefix_adapter = GraphPrefixAdapter(
            hidden_size=hidden_size,
            prefix_len=prefix_len,
            dropout=prefix_dropout,
        )
        self.graph_residual_norm = nn.LayerNorm(hidden_size)

        self.use_graph = True
        self.graph_prefix_len = prefix_len
        self.graph_dim = graph_dim
        self.align_loss_weight = graph_config.get("align_loss_weight", 0.0)
        self.latent_injection = graph_config.get("latent_injection", "residual")

    def _slice_graph_for_analysis(self, graph_batch, graph_edge_index, batch_index):
        node_indices = (graph_batch == batch_index).nonzero(as_tuple=False).view(-1)
        if node_indices.numel() == 0:
            return node_indices, torch.empty((2, 0), dtype=torch.long, device=graph_batch.device)

        edge_mask = (
            (graph_batch[graph_edge_index[0]] == batch_index)
            & (graph_batch[graph_edge_index[1]] == batch_index)
        )
        scoped_edges = graph_edge_index[:, edge_mask]
        remap = torch.full(
            (graph_batch.shape[0],),
            -1,
            dtype=torch.long,
            device=graph_batch.device,
        )
        remap[node_indices] = torch.arange(
            node_indices.numel(), dtype=torch.long, device=graph_batch.device
        )
        return node_indices, remap[scoped_edges]

    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        position_ids,
        graph_x=None,
        graph_edge_index=None,
        graph_batch=None,
        graph_role=None,
        analysis_trace: Optional[dict] = None,
        analysis_batch_index: int = 0,
        **kwargs,
    ):

        analysis_enabled = analysis_trace is not None

        if self.use_graph:
            if graph_x is None or graph_edge_index is None or graph_batch is None:
                raise ValueError(
                    "Graph conditioning requested but graph inputs are missing."
                )
            z_g, h_nodes = self.graph_encoder(graph_x, graph_edge_index, graph_batch)
            graph_embedding = self.graph_projector(z_g)
            node_embeddings = self.graph_projector(h_nodes)
            graph_prefix = (
                self.graph_prefix_adapter(graph_embedding)
                if self.graph_prefix_len > 0
                else None
            )
            graph_residual = self.graph_residual_norm(graph_embedding)

            if analysis_enabled:
                if analysis_batch_index < 0 or analysis_batch_index >= z_g.shape[0]:
                    raise IndexError(
                        f"analysis_batch_index={analysis_batch_index} is out of range for batch size {z_g.shape[0]}"
                    )
                node_indices, edge_index_local = self._slice_graph_for_analysis(
                    graph_batch, graph_edge_index, analysis_batch_index
                )
                role_local = (
                    graph_role[node_indices]
                    if graph_role is not None
                    else torch.zeros((node_indices.numel(),), dtype=torch.long, device=graph_batch.device)
                )
                analysis_trace["node_embeddings"] = (
                    node_embeddings[node_indices].detach().cpu()
                )
                analysis_trace["edge_index"] = edge_index_local.detach().cpu()
                analysis_trace["role"] = role_local.detach().cpu()
                analysis_trace["node_count"] = int(node_indices.numel())
                analysis_trace.setdefault("trajectory", [])
        else:
            graph_embedding = None
            graph_prefix = None
            graph_residual = None

        batch_size = input_ids.shape[0]
        inputs_embeds = self.embedding(input_ids)

        if graph_prefix is not None:
            inputs_embeds = torch.cat([graph_prefix, inputs_embeds], dim=1)
            prefix_mask = torch.ones(
                (batch_size, self.graph_prefix_len),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            prefix_positions = torch.arange(
                self.graph_prefix_len, device=position_ids.device
            ).unsqueeze(0)
            prefix_positions = prefix_positions.expand(batch_size, -1)
            position_ids = torch.cat([prefix_positions, position_ids + self.graph_prefix_len], dim=1)
            if labels is not None:
                prefix_labels = torch.full(
                    (batch_size, self.graph_prefix_len),
                    self.label_pad_token_id,
                    dtype=labels.dtype,
                    device=labels.device,
                )
                labels = torch.cat([prefix_labels, labels], dim=1)
            pad_prefix = torch.full(
                (batch_size, self.graph_prefix_len),
                self.pad_token_id,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            input_ids = torch.cat([pad_prefix, input_ids], dim=1)

        logits = []

        latent_indices = (
            input_ids == self.latent_token_id
        ).nonzero()  # (num_latent_tokens_in_the_batch, 2)

        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]  # bs, num_latent_tokens_in_the_instance (difference across the batch)

        max_n_latents = max([len(l) for l in latent_lists])

        next_compute_range = (0, input_ids.shape[1])

        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())
            # before the earliest latent token position

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

            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair
                new_state = hidden_states[
                    batch_idx, token_idx - 1 - hidden_states_offset, :
                ]
                if (
                    graph_residual is not None
                    and self.latent_injection == "residual"
                    and pass_idx == 0
                ):
                    new_state = new_state + graph_residual[batch_idx]

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

        if graph_embedding is not None and self.align_loss_weight > 0:
            pre_decode_hidden = outputs.hidden_states[-1][:, -1, :]
            norm_proj = F.normalize(graph_embedding, dim=-1)
            norm_hidden = F.normalize(pre_decode_hidden, dim=-1)
            align_loss = 1 - F.cosine_similarity(norm_proj, norm_hidden, dim=-1)
            align_loss = align_loss.mean()
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
            graph_embedding=graph_embedding,
        )

    def train(self):
        self.base_causallm.train()

    def eval(self):
        self.base_causallm.eval()

    def generate(
        self,
        input_ids,
        attention_mask,  # attention_mask is not used
        max_new_tokens=16,
        output_embedding=False,
        synced_gpus=False,
        graph_x=None,
        graph_edge_index=None,
        graph_batch=None,
        graph_role=None,
        analysis_trace: Optional[dict] = None,
        analysis_batch_index: int = 0,
        **kwargs
    ):

        self.gen_forward_cnt = 0

        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"

        if self.use_graph and (graph_x is None or graph_edge_index is None or graph_batch is None):
            raise ValueError(
                "Graph conditioning requires graph inputs during generation."
            )

        tokens = input_ids[0].detach().tolist()

        labels = input_ids.clone()  # placeholder. not used.
        outputs = self.forward(
            input_ids,
            torch.ones_like(input_ids, device=input_ids.device),
            labels,
            torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).reshape(1, -1),
            graph_x=graph_x,
            graph_edge_index=graph_edge_index,
            graph_batch=graph_batch,
            graph_role=graph_role,
            analysis_trace=analysis_trace,
            analysis_batch_index=analysis_batch_index,
        )
        inputs_embeds = outputs.inputs_embeds

        # get the first token using the current hidden state
        next_token = torch.argmax(outputs.logits[0, -1]).item()
        tokens.append(next_token)
        new_token_embed = self.embedding(
            torch.tensor(next_token, device=input_ids.device)
        ).view(1, 1, -1)
        new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)

        # get other tokens
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
            # in FSDP, the number of forward pass need to be the same across devices
            while (
                self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT
            ):  # leave some room for latent tokens
                self.gen_forward_cnt += 1
                _ = self.base_causallm(inputs_embeds=new_inputs_embeds)

        if output_embedding:
            # for analysis purpose
            return torch.tensor(tokens).view(1, -1), new_inputs_embeds

        else:
            return torch.tensor(tokens).view(1, -1)
