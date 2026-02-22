import json
from types import SimpleNamespace

import torch

from coconut import Coconut
from run import _freeze_base_llm_if_configured


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


def test_freeze_base_llm_leaves_projector_trainable(tmp_path):
    entity = torch.randn(4, 6)
    torch.save(entity, tmp_path / "entity_embeddings.pt")
    with (tmp_path / "metadata.json").open("w") as f:
        json.dump({"projector_in_dim": 6}, f)

    model = Coconut(
        DummyLM(hidden_size=8),
        latent_token_id=3,
        start_latent_id=4,
        end_latent_id=5,
        eos_token_id=1,
        pad_token_id=0,
        kge_config={
            "use_kge": True,
            "kge_artifact_root": str(tmp_path),
            "kge_entity_embeddings_file": "entity_embeddings.pt",
            "kge_metadata_file": "metadata.json",
        },
    )

    cfg = SimpleNamespace(freeze_base_llm=True, coconut=True, use_kge=True)
    frozen_count = _freeze_base_llm_if_configured(model, cfg)

    assert frozen_count > 0
    assert all(not p.requires_grad for p in model.base_causallm.parameters())
    assert any(p.requires_grad for p in model.kge_projector.parameters())
