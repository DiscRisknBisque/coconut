import json

import torch

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


def test_align_loss_disabled_with_zero_weight(tmp_path):
    torch.save(torch.randn(4, 6), tmp_path / "entity_embeddings.pt")
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
            "align_loss_weight": 0.0,
        },
    )

    input_ids = torch.tensor([[1, 2, 1]])
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    position_ids = torch.arange(input_ids.size(1)).unsqueeze(0)

    outputs = model(
        input_ids,
        attention_mask,
        labels,
        position_ids,
        anchor_entity_ids=torch.tensor([[0, -1, -1]]),
        anchor_token_spans=torch.tensor([[[1, 2], [0, 0], [0, 0]]]),
        anchor_mask=torch.tensor([[True, False, False]]),
    )

    assert outputs.align_loss is None
    assert outputs.loss.ndim == 0
