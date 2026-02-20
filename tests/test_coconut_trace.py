import pytest
import torch

pytest.importorskip("torch_geometric")

from coconut import Coconut


class DummyTransformer(torch.nn.Module):
    def __init__(self, embedding: torch.nn.Embedding):
        super().__init__()
        self._embedding = embedding

    def get_input_embeddings(self):
        return self._embedding


class DummyLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 16, hidden_size: int = 8):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.transformer = DummyTransformer(self.embedding)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.embedding

    def forward(
        self,
        inputs_embeds,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_hidden_states=False,
        **kwargs,
    ):
        hidden_states = inputs_embeds
        logits = self.lm_head(hidden_states)
        if past_key_values is None:
            seq = hidden_states.shape[1]
            zeros = torch.zeros(
                (hidden_states.shape[0], 1, seq, hidden_states.shape[-1]),
                device=hidden_states.device,
            )
            past_key_values = [(zeros, zeros)]
        return type(
            "DummyOutput",
            (),
            {
                "logits": logits,
                "hidden_states": [hidden_states],
                "past_key_values": past_key_values,
            },
        )()


def test_trace_capture_is_opt_in_and_collects_latent_trajectory():
    base = DummyLM()
    model = Coconut(
        base,
        latent_token_id=3,
        start_latent_id=4,
        end_latent_id=5,
        eos_token_id=1,
        pad_token_id=0,
        graph_config={
            "use_graph": True,
            "graph_encoder": "gcn2",
            "graph_dim": 6,
            "graph_hidden_dim": 6,
            "graph_input_dim": 6,
            "graph_prefix_len": 0,
            "align_loss_weight": 0.0,
            "latent_injection": "residual",
        },
    )

    input_ids = torch.tensor([[1, 3, 2]])
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    position_ids = torch.arange(input_ids.size(1)).unsqueeze(0)

    graph_x = torch.randn(3, 6)
    graph_edge_index = torch.tensor([[0, 1, 1], [1, 0, 2]], dtype=torch.long)
    graph_batch = torch.zeros(3, dtype=torch.long)
    graph_role = torch.tensor([0, 1, 2], dtype=torch.long)

    outputs_no_trace = model(
        input_ids,
        attention_mask,
        labels,
        position_ids,
        graph_x=graph_x,
        graph_edge_index=graph_edge_index,
        graph_batch=graph_batch,
        graph_role=graph_role,
    )
    assert outputs_no_trace.loss.ndim == 0

    trace = {}
    outputs_trace = model(
        input_ids,
        attention_mask,
        labels,
        position_ids,
        graph_x=graph_x,
        graph_edge_index=graph_edge_index,
        graph_batch=graph_batch,
        graph_role=graph_role,
        analysis_trace=trace,
        analysis_batch_index=0,
    )

    assert outputs_trace.loss.ndim == 0
    assert "node_embeddings" in trace
    assert "trajectory" in trace
    assert "edge_index" in trace
    assert "role" in trace
    assert trace["node_embeddings"].shape[0] == 3
    assert trace["node_embeddings"].shape[1] == base.embedding.embedding_dim
    assert len(trace["trajectory"]) == 1

