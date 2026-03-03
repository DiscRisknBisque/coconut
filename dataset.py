# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

from kge_utils import ANCHOR_FIELDS, build_anchor_payload, load_entity_to_id


def _infer_split_from_path(path: Path) -> str:
    stem = path.stem.lower()
    for candidate in ("train", "valid", "val", "test", "dev"):
        if stem.endswith(candidate):
            return "valid" if candidate == "val" else candidate
    raise ValueError(f"Unable to infer split name from path '{path}'")


def get_dataset(
    path,
    tokenizer,
    max_size=1000000000,
    use_kge: bool = False,
    kge_entity_to_id_path: Optional[str] = None,
    kge_anchor_policy: str = "query_anchors",
):
    if use_kge and kge_anchor_policy != "query_anchors":
        raise ValueError(
            f"Unsupported kge_anchor_policy='{kge_anchor_policy}'. Expected 'query_anchors'."
        )

    if use_kge:
        if not kge_entity_to_id_path:
            raise ValueError(
                "KGE usage requested but kge_entity_to_id_path was not provided."
            )
        entity_to_id = load_entity_to_id(kge_entity_to_id_path)
    else:
        entity_to_id = None

    def tokenize_sample(sample):
        question_text = sample["question"] + "\n"
        question_tokenized = tokenizer.encode(question_text, add_special_tokens=True)
        steps_tokenized = [
            tokenizer.encode(s + "\n", add_special_tokens=False)
            for s in sample["steps"]
        ]
        answer_tokenized = tokenizer.encode(
            "### " + sample["answer"], add_special_tokens=False
        ) + [tokenizer.eos_token_id]

        if use_kge:
            anchor_entity_ids, anchor_token_spans, anchor_mask = build_anchor_payload(
                sample=sample,
                tokenizer=tokenizer,
                question_text=question_text,
                question_tokenized=question_tokenized,
                entity_to_id=entity_to_id,
            )
        else:
            anchor_entity_ids = [-1] * len(ANCHOR_FIELDS)
            anchor_token_spans = [[0, 0] for _ in ANCHOR_FIELDS]
            anchor_mask = [False] * len(ANCHOR_FIELDS)

        sample = {
            "question_tokenized": question_tokenized,
            "steps_tokenized": steps_tokenized,
            "answer_tokenized": answer_tokenized,
            "idx": sample["idx"],
            "anchor_entity_ids": anchor_entity_ids,
            "anchor_token_spans": anchor_token_spans,
            "anchor_mask": anchor_mask,
        }
        return sample

    path_obj = Path(path)
    _infer_split_from_path(path_obj)

    data = json.load(open(path))[:max_size]
    data = [{**d, "idx": idx} for idx, d in enumerate(data)]

    keys = data[0].keys()
    dataset = Dataset.from_dict({k: [d[k] for d in data] for k in keys})

    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            processed_dataset = [
                dataset.map(
                    tokenize_sample, remove_columns=list(dataset.features), num_proc=32
                )
            ]
        else:
            processed_dataset = [None]
        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]

    else:
        dataset = dataset.map(
            tokenize_sample, remove_columns=list(dataset.features), num_proc=32
        )

    d = data[0]
    complete = d["question"] + "\n" + "\n".join(d["steps"]) + "\n### " + d["answer"]
    complete_tokenized = tokenizer.encode(complete, add_special_tokens=True) + [
        tokenizer.eos_token_id
    ]
    assert (
        complete_tokenized
        == dataset[0]["question_tokenized"]
        + list(itertools.chain.from_iterable(dataset[0]["steps_tokenized"]))
        + dataset[0]["answer_tokenized"]
    )

    return dataset


@dataclass
class MyCollator:
    tokenizer: PreTrainedTokenizerBase
    latent_id: Optional[int] = None
    label_pad_token_id: Optional[int] = -100
    use_kge: bool = False

    def __call__(self, features, return_tensors=None):
        assert self.tokenizer.padding_side == "right"

        earliest_latent = []
        for feature in features:
            if self.use_kge:
                feature.setdefault("anchor_entity_ids", [-1] * len(ANCHOR_FIELDS))
                feature.setdefault(
                    "anchor_token_spans",
                    [[0, 0] for _ in ANCHOR_FIELDS],
                )
                feature.setdefault("anchor_mask", [False] * len(ANCHOR_FIELDS))
            if self.latent_id in feature["input_ids"]:
                earliest_latent.append(feature["input_ids"].index(self.latent_id))

        if len(earliest_latent) > 0:
            latest_earliest_latent = max(earliest_latent)
            for feature in features:
                if self.latent_id in feature["input_ids"]:
                    n_tok_pad = latest_earliest_latent - feature["input_ids"].index(
                        self.latent_id
                    )
                else:
                    n_tok_pad = 0
                feature["position_ids"] = [0] * n_tok_pad + list(
                    range(len(feature["input_ids"]))
                )
                feature["input_ids"] = [
                    self.tokenizer.pad_token_id
                ] * n_tok_pad + feature["input_ids"]
                if "labels" in feature:
                    feature["labels"] = [self.label_pad_token_id] * n_tok_pad + feature[
                        "labels"
                    ]
                feature["attention_mask"] = [0] * n_tok_pad + feature["attention_mask"]

                if self.use_kge and n_tok_pad > 0:
                    shifted_spans = []
                    for span, valid in zip(
                        feature["anchor_token_spans"], feature["anchor_mask"]
                    ):
                        if valid:
                            shifted_spans.append([span[0] + n_tok_pad, span[1] + n_tok_pad])
                        else:
                            shifted_spans.append([0, 0])
                    feature["anchor_token_spans"] = shifted_spans

        return_tensors = "pt"

        label_name = "label" if "label" in features[0].keys() else "labels"

        non_label_position_features = [
            {
                k: v
                for k, v in feature.items()
                if k
                not in {
                    label_name,
                    "position_ids",
                    "anchor_entity_ids",
                    "anchor_token_spans",
                    "anchor_mask",
                }
            }
            for feature in features
        ]

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_label_position_features,
            padding=True,
            pad_to_multiple_of=None,
            return_tensors=return_tensors,
        )

        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )
        if labels is not None and all(label is None for label in labels):
            labels = None
        position_ids = (
            [feature["position_ids"] for feature in features]
            if "position_ids" in features[0].keys()
            else None
        )

        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            batch["labels"] = [
                label + [self.label_pad_token_id] * (max_label_length - len(label))
                for label in labels
            ]
            batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)

        if position_ids is not None:
            max_pos_length = max(len(l) for l in position_ids)
            batch["position_ids"] = [
                position_id + [0] * (max_pos_length - len(position_id))
                for position_id in position_ids
            ]
            batch["position_ids"] = torch.tensor(
                batch["position_ids"], dtype=torch.int64
            )

        if self.use_kge:
            batch["anchor_entity_ids"] = torch.tensor(
                [feature["anchor_entity_ids"] for feature in features],
                dtype=torch.long,
            )
            batch["anchor_token_spans"] = torch.tensor(
                [feature["anchor_token_spans"] for feature in features],
                dtype=torch.long,
            )
            batch["anchor_mask"] = torch.tensor(
                [feature["anchor_mask"] for feature in features],
                dtype=torch.bool,
            )

        return batch


def get_question_latent_dataset(
    scheduled_stage,
    base_dataset_valid,
    configs,
    start_id,
    latent_id,
    end_id,
    no_special_marker=False,
):
    def process_dataset(sample):
        if configs.pad_latent_to_max:
            max_latent_stage = configs.max_latent_stage
        else:
            max_latent_stage = min(
                configs.max_latent_stage, len(sample["steps_tokenized"])
            )

        k = min(max_latent_stage, scheduled_stage)
        k *= configs.c_thought

        tokens = (
            sample["question_tokenized"]
            + ([] if no_special_marker else [start_id])
            + [latent_id] * k
            + ([] if no_special_marker else [end_id])
        )

        return {
            "input_ids": tokens,
            "idx": sample["idx"],
            "attention_mask": [1] * len(tokens),
            "position_ids": list(range(len(tokens))),
            "anchor_entity_ids": sample.get("anchor_entity_ids"),
            "anchor_token_spans": sample.get("anchor_token_spans"),
            "anchor_mask": sample.get("anchor_mask"),
        }

    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            processed_dataset = base_dataset_valid.map(
                process_dataset,
                remove_columns=list(base_dataset_valid.features),
                num_proc=32,
            )
            processed_dataset = [processed_dataset]
        else:
            processed_dataset = [None]
        dist.broadcast_object_list(processed_dataset, src=0)
        return processed_dataset[0]
    else:
        return base_dataset_valid.map(
            process_dataset,
            remove_columns=list(base_dataset_valid.features),
            num_proc=32,
        )


def get_cot_latent_dataset(
    scheduled_stage,
    base_dataset,
    configs,
    start_id,
    latent_id,
    end_id,
    no_special_marker=False,
    shuffle=False,
):
    n_additional_tokens = 0 if no_special_marker else 2

    def process_dataset(sample):
        if random.random() < configs.uniform_prob:
            scheduled_stage_to_train = random.choice(
                list(range(len(sample["steps_tokenized"]) + 1))
            )
        else:
            scheduled_stage_to_train = scheduled_stage

        if scheduled_stage_to_train > configs.max_latent_stage:
            n_skip_steps = 10000
            if configs.pad_latent_to_max:
                n_latent_tokens = configs.max_latent_stage
            else:
                n_latent_tokens = min(
                    len(sample["steps_tokenized"]), configs.max_latent_stage
                )

        else:
            n_skip_steps, n_latent_tokens = (
                scheduled_stage_to_train,
                scheduled_stage_to_train,
            )

        if configs.no_cot:
            n_skip_steps = 100
            n_latent_tokens = 0

        n_latent_tokens *= configs.c_thought

        tokens = (
            sample["question_tokenized"]
            + ([] if no_special_marker else [start_id])
            + [latent_id] * n_latent_tokens
            + ([] if no_special_marker else [end_id])
            + list(
                itertools.chain.from_iterable(sample["steps_tokenized"][n_skip_steps:])
            )
            + sample["answer_tokenized"]
        )

        return {
            "input_ids": tokens,
            "labels": [-100]
            * (
                len(sample["question_tokenized"])
                + n_latent_tokens
                + n_additional_tokens
            )
            + tokens[
                n_latent_tokens
                + n_additional_tokens
                + len(sample["question_tokenized"]) :
            ],
            "attention_mask": [1] * len(tokens),
            "idx": sample["idx"],
            "position_ids": list(range(len(tokens))),
            "anchor_entity_ids": sample.get("anchor_entity_ids"),
            "anchor_token_spans": sample.get("anchor_token_spans"),
            "anchor_mask": sample.get("anchor_mask"),
        }

    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            processed_dataset = base_dataset.map(
                process_dataset, remove_columns=list(base_dataset.features), num_proc=32
            )
            if shuffle:
                processed_dataset = processed_dataset.shuffle()
            processed_dataset = [processed_dataset]
        else:
            processed_dataset = [None]
        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]

    else:
        processed_dataset = base_dataset.map(
            process_dataset, remove_columns=list(base_dataset.features), num_proc=32
        )
        if shuffle:
            processed_dataset = processed_dataset.shuffle()
        dataset = processed_dataset

    return dataset
