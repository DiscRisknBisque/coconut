from pathlib import Path

import pytest
import torch

pytest.importorskip("torch_geometric")

from coconut import Coconut
from graph_encoder import GraphEncoder
from preprocessing.graph_utils import ROLE_TO_ID, save_graph_sidecars


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
        return type("DummyOutput", (), {
            "logits": logits,
            "hidden_states": [hidden_states],
            "past_key_values": past_key_values,
        })()


def test_graph_encoder_shapes():
    encoder = GraphEncoder(d_in=6, d_hidden=4, d_out=3)
    x = torch.randn(5, 6)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    batch = torch.zeros(5, dtype=torch.long)

    z_g, h_nodes = encoder(x, edge_index, batch)

    assert z_g.shape == (1, 3)
    assert h_nodes.shape == (5, 3)


def test_sidecar_roundtrip(tmp_path):
    dataset = [{"idx": 0, "question": "alpha beta"}]

    def extractor(item):
        return [("question", item["question"])], []

    manifest = save_graph_sidecars(dataset, tmp_path, "train", extractor, dim=4)
    manifest_path = tmp_path / "manifest_train.json"

    assert manifest_path.exists()
    saved_path = Path(manifest["0"])
    data = torch.load(saved_path)

    assert data["x"].shape[1] == 4 + len(ROLE_TO_ID)
    assert data["edge_index"].shape[0] == 2
    assert data["role"].shape[0] == data["x"].shape[0]


def test_align_loss_disabled_without_graph():
    base = DummyLM()
    model = Coconut(
        base,
        latent_token_id=3,
        start_latent_id=4,
        end_latent_id=5,
        eos_token_id=1,
        pad_token_id=0,
        graph_config={"use_graph": False},
    )

    input_ids = torch.tensor([[1, 2, 1]])
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    position_ids = torch.arange(input_ids.size(1)).unsqueeze(0)

    outputs = model(input_ids, attention_mask, labels, position_ids)

    assert outputs.align_loss is None
    assert outputs.graph_embedding is None
    assert outputs.loss.ndim == 0

