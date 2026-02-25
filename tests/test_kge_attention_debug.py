import json
from pathlib import Path

import torch

from analysis.kge_attention_debug import debug_kge_attention_for_batch
from coconut import Coconut


class DummyTransformer(torch.nn.Module):
    def __init__(self, embedding: torch.nn.Embedding):
        super().__init__()
        self._embedding = embedding

    def get_input_embeddings(self):
        return self._embedding


class DummyLMWithAttention(torch.nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_size: int = 8):
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
        output_attentions=False,
        return_dict=False,
        **kwargs,
    ):
        hidden_states = inputs_embeds
        logits = self.lm_head(hidden_states)

        batch, seq, hidden = hidden_states.shape
        if past_key_values is None:
            zeros = torch.zeros((batch, 1, seq, hidden), device=hidden_states.device)
            past_key_values = [(zeros, zeros)]

        attentions = None
        if output_attentions:
            attn = torch.full(
                (batch, 1, seq, seq),
                fill_value=1.0 / max(seq, 1),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            # Make the final token attend strongly to position 1 and weakly elsewhere.
            if seq >= 4:
                attn[:, 0, -1, :] = torch.tensor(
                    [0.05, 0.60, 0.25, 0.10] + [0.0] * (seq - 4),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
            attentions = [attn]

        output = type(
            "DummyOutput",
            (),
            {
                "logits": logits,
                "hidden_states": [hidden_states] if output_hidden_states else None,
                "past_key_values": past_key_values,
                "attentions": attentions,
            },
        )()
        return output


def _write_kge_artifacts(tmp_path: Path):
    entity = torch.randn(4, 6)
    torch.save(entity, tmp_path / "entity_embeddings.pt")
    with (tmp_path / "metadata.json").open("w") as f:
        json.dump({"projector_in_dim": 6}, f)


def _build_model_and_batch(tmp_path: Path):
    _write_kge_artifacts(tmp_path)
    base = DummyLMWithAttention(hidden_size=8)
    model = Coconut(
        base,
        latent_token_id=30,
        start_latent_id=29,
        end_latent_id=28,
        eos_token_id=1,
        pad_token_id=0,
        kge_config={
            "use_kge": True,
            "kge_artifact_root": str(tmp_path),
            "kge_entity_embeddings_file": "entity_embeddings.pt",
            "kge_metadata_file": "metadata.json",
            "kge_projector_hidden": 10,
            "kge_projector_activation": "gelu",
            "kge_projector_layernorm": True,
            "align_loss_weight": 0.0,
            "latent_injection": "residual",
        },
    )

    input_ids = torch.tensor([[7, 8, 9, 10]])
    batch = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
        "position_ids": torch.arange(input_ids.shape[1]).unsqueeze(0),
        "anchor_entity_ids": torch.tensor([[0, -1, -1]]),
        "anchor_token_spans": torch.tensor([[[1, 2], [0, 0], [0, 0]]]),
        "anchor_mask": torch.tensor([[True, False, False]]),
    }
    return model, batch


def test_debug_kge_attention_reports_anchor_mass_for_valid_span(tmp_path):
    model, batch = _build_model_and_batch(tmp_path)

    result = debug_kge_attention_for_batch(model, batch, print_report=False)

    assert result["anchor_attention_mass"] > 0
    assert result["anchor_attention_pct"] > 0
    assert result["valid_anchor_count"] == 1
    assert "anchors" in result
    assert "top_attention_tokens" in result


def test_debug_kge_attention_handles_no_valid_anchors_cleanly(tmp_path):
    model, batch = _build_model_and_batch(tmp_path)
    batch["anchor_mask"] = torch.tensor([[False, False, False]])

    result = debug_kge_attention_for_batch(model, batch, print_report=False)

    assert result["valid_anchor_count"] == 0
    assert result["anchor_attention_mass"] == 0.0
    assert any("No valid anchors" in note for note in result["notes"])


def test_debug_kge_attention_ignores_invalid_spans(tmp_path):
    model, batch = _build_model_and_batch(tmp_path)
    batch["anchor_token_spans"] = torch.tensor([[[3, 2], [0, 0], [0, 0]]])
    batch["anchor_mask"] = torch.tensor([[True, False, False]])

    result = debug_kge_attention_for_batch(model, batch, print_report=False)

    assert result["valid_anchor_count"] == 0
    assert result["anchor_attention_mass"] == 0.0
    assert len(result["skipped_anchors"]) == 1
    assert "invalid span" in result["skipped_anchors"][0]["reason"]


def test_debug_kge_attention_prints_human_readable_report(tmp_path, capsys):
    model, batch = _build_model_and_batch(tmp_path)

    debug_kge_attention_for_batch(model, batch, print_report=True)
    captured = capsys.readouterr().out

    assert "--- KGE ATTENTION DIAGNOSTIC ---" in captured
    assert "Total Anchor/KGE Attention" in captured
    assert any(label in captured for label in ["[WARNING]", "[MODERATE]", "[HEALTHY]"])
