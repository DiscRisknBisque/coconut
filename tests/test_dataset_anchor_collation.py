import torch

from dataset import MyCollator


class DummyTokenizer:
    padding_side = "right"
    pad_token_id = 0

    def __init__(self):
        self.deprecation_warnings = {}

    def pad(
        self,
        encoded_inputs,
        padding=True,
        max_length=None,
        pad_to_multiple_of=None,
        return_tensors=None,
        **kwargs,
    ):
        max_len = max(len(item["input_ids"]) for item in encoded_inputs)
        input_ids = []
        attention_mask = []
        for item in encoded_inputs:
            ids = item["input_ids"]
            mask = item["attention_mask"]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad_len)
            attention_mask.append(mask + [0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def test_anchor_collation_shapes_and_left_pad_shift():
    tokenizer = DummyTokenizer()
    collator = MyCollator(
        tokenizer=tokenizer,
        latent_id=3,
        label_pad_token_id=-100,
        use_kge=True,
    )

    features = [
        {
            "input_ids": [10, 11, 3, 12],
            "labels": [10, 11, 3, 12],
            "attention_mask": [1, 1, 1, 1],
            "position_ids": [0, 1, 2, 3],
            "anchor_entity_ids": [5, 6, -1],
            "anchor_token_spans": [[0, 1], [1, 2], [0, 0]],
            "anchor_mask": [True, True, False],
        },
        {
            "input_ids": [20, 3, 21],
            "labels": [20, 3, 21],
            "attention_mask": [1, 1, 1],
            "position_ids": [0, 1, 2],
            "anchor_entity_ids": [7, -1, -1],
            "anchor_token_spans": [[0, 1], [0, 0], [0, 0]],
            "anchor_mask": [True, False, False],
        },
    ]

    batch = collator(features)

    assert batch["anchor_entity_ids"].shape == (2, 3)
    assert batch["anchor_token_spans"].shape == (2, 3, 2)
    assert batch["anchor_mask"].dtype == torch.bool

    assert batch["anchor_token_spans"][1, 0].tolist() == [1, 2]
    assert batch["anchor_token_spans"][0, 0].tolist() == [0, 1]
