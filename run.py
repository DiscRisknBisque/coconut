# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import argparse
import functools
import gc
import json
import math
import os
import sys
from contextlib import nullcontext

import torch
import torch.distributed
import torch.optim as optim
import torch.nn.utils as nn_utils
import wandb
import yaml
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from coconut import Coconut
from dataset import MyCollator, get_cot_latent_dataset, get_dataset, get_question_latent_dataset
from utils import Config, set_seed


def _freeze_base_llm_if_configured(model, configs) -> int:
    if not (
        getattr(configs, "freeze_base_llm", True)
        and getattr(configs, "coconut", False)
        and getattr(configs, "use_kge", False)
    ):
        return 0
    if not hasattr(model, "base_causallm"):
        return 0

    frozen = 0
    for param in model.base_causallm.parameters():
        if param.requires_grad:
            param.requires_grad = False
            frozen += param.numel()
    return frozen


def _trainable_parameters(module):
    return [p for p in module.parameters() if p.requires_grad]


def _validate_lr(value, field_name: str) -> float:
    lr = float(value)
    if lr <= 0:
        raise ValueError(f"{field_name} must be > 0, got {value}")
    return lr


def _resolve_group_lrs(*, lr, lr_base_llm=None, lr_projection_mlp=None):
    default_lr = _validate_lr(lr, "lr")
    base_lr = (
        _validate_lr(lr_base_llm, "lr_base_llm")
        if lr_base_llm is not None
        else default_lr
    )
    projection_lr = (
        _validate_lr(lr_projection_mlp, "lr_projection_mlp")
        if lr_projection_mlp is not None
        else default_lr
    )
    return default_lr, base_lr, projection_lr


def _build_optimizer_param_groups(module, *, lr, lr_base_llm=None, lr_projection_mlp=None):
    trainable_named = [(name, p) for name, p in module.named_parameters() if p.requires_grad]
    if len(trainable_named) == 0:
        raise RuntimeError("No trainable parameters found for optimizer initialization")

    default_lr, base_lr, projection_lr = _resolve_group_lrs(
        lr=lr,
        lr_base_llm=lr_base_llm,
        lr_projection_mlp=lr_projection_mlp,
    )
    use_split_lrs = lr_base_llm is not None or lr_projection_mlp is not None

    if not use_split_lrs:
        return (
            [{"name": "all", "params": [p for _, p in trainable_named], "lr": default_lr}],
            {"all": trainable_named},
            [],
        )

    grouped_named = {"base_llm": [], "projection_mlp": [], "other": []}
    for name, param in trainable_named:
        if "kge_projector" in name or "kge_residual_norm" in name:
            grouped_named["projection_mlp"].append((name, param))
        elif "base_causallm" in name:
            grouped_named["base_llm"].append((name, param))
        else:
            grouped_named["other"].append((name, param))

    seen_param_ids = set()
    for group_items in grouped_named.values():
        for _, param in group_items:
            param_id = id(param)
            if param_id in seen_param_ids:
                raise RuntimeError("Parameter was assigned to multiple optimizer groups")
            seen_param_ids.add(param_id)

    grouped_lr = {
        "base_llm": base_lr,
        "projection_mlp": projection_lr,
        "other": default_lr,
    }
    optimizer_groups = []
    optimizer_named_params = {}
    for group_name in ("base_llm", "projection_mlp", "other"):
        named_items = grouped_named[group_name]
        if len(named_items) == 0:
            continue
        optimizer_groups.append(
            {
                "name": group_name,
                "params": [param for _, param in named_items],
                "lr": grouped_lr[group_name],
            }
        )
        optimizer_named_params[group_name] = named_items

    missing_overrides = []
    if lr_base_llm is not None and len(grouped_named["base_llm"]) == 0:
        missing_overrides.append("base_llm")
    if lr_projection_mlp is not None and len(grouped_named["projection_mlp"]) == 0:
        missing_overrides.append("projection_mlp")

    return optimizer_groups, optimizer_named_params, missing_overrides


def _group_grad_norm(parameters, device):
    total = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        grad_sq = grad.pow(2).sum()
        total = grad_sq if total is None else total + grad_sq
    if total is None:
        return None
    return total.sqrt().to(device=device)


def _group_param_norm(parameters, device):
    total = None
    for parameter in parameters:
        param_sq = parameter.detach().float().pow(2).sum()
        total = param_sq if total is None else total + param_sq
    if total is None:
        return torch.tensor(0.0, device=device)
    return total.sqrt().to(device=device)


def _optimizer_update_steps(num_batches: int, gradient_accumulation_steps: int) -> int:
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be > 0")
    if num_batches <= 0:
        return 0
    return math.ceil(num_batches / gradient_accumulation_steps)


def _remaining_epochs_in_current_stage(
    *,
    epoch: int,
    num_epochs: int,
    epochs_per_stage: int,
    single_stage_schedule: bool,
) -> int:
    if epoch >= num_epochs:
        return 0
    if single_stage_schedule:
        return num_epochs - epoch
    if epochs_per_stage <= 0:
        raise ValueError("epochs_per_stage must be > 0 when using staged training")
    stage_idx = epoch // epochs_per_stage
    stage_end_epoch = min(num_epochs, (stage_idx + 1) * epochs_per_stage)
    return max(stage_end_epoch - epoch, 0)


def _scheduler_step_counts(
    *,
    num_batches: int,
    gradient_accumulation_steps: int,
    epoch: int,
    num_epochs: int,
    reset_optimizer: bool,
    epochs_per_stage: int,
    single_stage_schedule: bool,
    lr_warmup_ratio: float,
):
    updates_per_epoch = _optimizer_update_steps(num_batches, gradient_accumulation_steps)
    if reset_optimizer:
        remaining_epochs = _remaining_epochs_in_current_stage(
            epoch=epoch,
            num_epochs=num_epochs,
            epochs_per_stage=epochs_per_stage,
            single_stage_schedule=single_stage_schedule,
        )
    else:
        remaining_epochs = max(num_epochs - epoch, 0)

    total_steps = max(updates_per_epoch * remaining_epochs, 0)
    warmup_ratio = min(max(float(lr_warmup_ratio), 0.0), 1.0)
    warmup_steps = min(max(int(total_steps * warmup_ratio), 0), total_steps)
    return updates_per_epoch, total_steps, warmup_steps


def _resolve_distributed_env(local_rank: int):
    """Pick the best distributed backend and compute device for this machine."""
    if torch.cuda.is_available():
        backend = "nccl"
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(local_rank)
    else:
        backend = "gloo"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    return backend, device


def _get_transformer_wrap_policy(model):
    """Return FSDP auto-wrap policy matching the actual base model's layer class."""
    from transformers.models.gpt2 import GPT2LMHeadModel as _GPT2LMHeadModel
    base = getattr(model, "base_causallm", model)
    if isinstance(base, _GPT2LMHeadModel):
        layer_cls = {GPT2Block}
    else:
        try:
            from transformers.models.llama.modeling_llama import LlamaDecoderLayer
            layer_cls = {LlamaDecoderLayer}
        except ImportError:
            raise RuntimeError(
                f"Unrecognised base model type {type(base)}; "
                "add its transformer layer class to _get_transformer_wrap_policy."
            )
    return functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=layer_cls)


@record
def main():
    parser = argparse.ArgumentParser(description="coconut")
    parser.add_argument("config_file")
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    backend, device = _resolve_distributed_env(local_rank)
    dist.init_process_group(backend)

    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)

    if rank == 0:
        print("Config:", config_dict)

    configs = Config(config_dict)
    set_seed(configs.seed)

    default_kge_config = {
        "use_kge": False,
        "kge_artifact_root": None,
        "kge_entity_embeddings_file": "entity_embeddings.pt",
        "kge_relation_embeddings_file": "relation_embeddings.pt",
        "kge_entity_to_id_file": "entity_to_id.json",
        "kge_metadata_file": "metadata.json",
        "kge_anchor_policy": "query_anchors",
        "kge_projector_hidden": None,
        "kge_projector_num_hidden_layers": 1,
        "kge_projector_activation": "gelu",
        "kge_projector_layernorm": True,
        "align_loss_weight": 0.0,
        "latent_injection": "residual",
        "freeze_base_llm": True,
    }

    for key, value in default_kge_config.items():
        if not hasattr(configs, key):
            setattr(configs, key, value)

    default_train_stability_config = {
        "lr_scheduler": "none",
        "lr_warmup_ratio": 0.0,
        "max_grad_norm": None,
        "lr_base_llm": None,
        "lr_projection_mlp": None,
        "wandb_log_group_grad_norm": False,
        "wandb_log_group_param_norm": False,
        "wandb_group_norm_log_interval": 50,
        "wandb_log_training_data": True,
        "wandb_training_data_max_examples": 2,
        "wandb_training_data_max_tokens_per_example": 256,
        "wandb_training_data_max_chars": 20000,
    }
    for key, value in default_train_stability_config.items():
        if not hasattr(configs, key):
            setattr(configs, key, value)

    if configs.use_kge and not configs.kge_artifact_root:
        raise ValueError(
            "KGE conditioning enabled but 'kge_artifact_root' is not configured."
        )

    kge_entity_to_id_path = None
    if configs.use_kge:
        kge_entity_to_id_path = os.path.join(
            configs.kge_artifact_root, configs.kge_entity_to_id_file
        )
        if not os.path.exists(kge_entity_to_id_path):
            raise FileNotFoundError(
                f"Missing entity mapping file for KGE mode: {kge_entity_to_id_path}"
            )

    save_dir = os.path.join(configs.save_path, configs.name)

    if not os.path.exists(save_dir) and rank == 0:
        os.makedirs(save_dir)

    torch.distributed.barrier()
    cur_ckpts = os.listdir(save_dir)

    if len(cur_ckpts) > 0 and not configs.only_eval:
        if rank == 0:
            print(
                "Warning: found previous run and gonna resume from that. "
                "the inputted `resume` argument is ignored!"
            )

        checkpoints = [f for f in cur_ckpts if f.startswith("checkpoint_")]
        checkpoints.sort(key=lambda x: int(x.split("_")[1]))

        latest_checkpoint = checkpoints[-1] if checkpoints else None
        configs.resume = int(latest_checkpoint.split("_")[1])
        load_dir = os.path.join(configs.save_path, configs.name, latest_checkpoint)

        configs.load_model_path = load_dir
        print(f"Loading from previous run epoch_{configs.resume}!")

    elif configs.resume != 0:
        if configs.load_model_path == "None":
            print(
                f"Warning: you want to skip the first {configs.resume} "
                "but you are not loading any existing checkpoint!"
            )
        print(
            f"Loading from {configs.load_model_path} and skip the first {configs.resume} epochs"
        )

    model = AutoModelForCausalLM.from_pretrained(configs.model_id)
    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    kge_config = {
        "use_kge": configs.use_kge,
        "kge_artifact_root": configs.kge_artifact_root,
        "kge_entity_embeddings_file": configs.kge_entity_embeddings_file,
        "kge_relation_embeddings_file": configs.kge_relation_embeddings_file,
        "kge_entity_to_id_file": configs.kge_entity_to_id_file,
        "kge_metadata_file": configs.kge_metadata_file,
        "kge_anchor_policy": configs.kge_anchor_policy,
        "kge_projector_hidden": configs.kge_projector_hidden,
        "kge_projector_num_hidden_layers": configs.kge_projector_num_hidden_layers,
        "kge_projector_activation": configs.kge_projector_activation,
        "kge_projector_layernorm": configs.kge_projector_layernorm,
        "align_loss_weight": configs.align_loss_weight,
        "latent_injection": configs.latent_injection,
    }

    loaded = False

    if configs.load_model_path != "None":
        saved_weights = torch.load(
            configs.load_model_path, map_location=device
        )

        if configs.coconut and not any(
            [k.startswith("base_causallm") for k in saved_weights.keys()]
        ):
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))

        elif not configs.coconut and any(
            [k.startswith("base_causallm") for k in saved_weights.keys()]
        ):
            raise ValueError("Cannot load coconut model weights into a causallm model")

        elif configs.coconut and any(
            [k.startswith("base_causallm") for k in saved_weights.keys()]
        ):
            pass

        else:
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))

    if not (configs.cot or configs.no_thoughts or configs.no_cot):
        model.resize_token_embeddings(len(tokenizer))
        embeddings = model.get_input_embeddings()
        target_id = tokenizer.convert_tokens_to_ids("<<")
        for token_id in [latent_id, start_id, end_id]:
            target_embedding = embeddings.weight.data[target_id]
            embeddings.weight.data[token_id] = target_embedding
            lm_head = model.lm_head
            lm_head.weight.data[token_id] = lm_head.weight.data[target_id]

    if configs.no_thoughts:
        configs.c_thought = 0
        configs.coconut = False

    if configs.coconut:
        model = Coconut(
            model,
            latent_id,
            start_id,
            end_id,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
            kge_config=kge_config,
        )

    if configs.load_model_path != "None" and not loaded:
        print(model.load_state_dict(saved_weights, strict=False))

    frozen_params = _freeze_base_llm_if_configured(model, configs)
    if rank == 0 and frozen_params > 0:
        print(f"Froze {frozen_params} base-LLM parameters for KGE alignment phase.")

    print(f"Running on rank={rank}, world_size={world_size}, device={device}")
    model = model.to(device)

    wrap_policy = _get_transformer_wrap_policy(model)

    if configs.bf16:
        model.to(torch.bfloat16)

    if torch.cuda.is_available():
        if configs.only_eval:
            parallel_model = DDP(model, device_ids=[local_rank])
        else:
            parallel_model = FSDP(
                model,
                auto_wrap_policy=wrap_policy,
                device_id=device,
                # KGE mode freezes base LLM params while keeping projector params trainable.
                # Without use_orig_params, FSDP requires uniform requires_grad within each flattened handle.
                use_orig_params=True,
            )
    else:
        parallel_model = DDP(model)
    use_pin_memory = torch.cuda.is_available()

    del model

    if rank == 0:
        print(parallel_model)

    question_val = [d["question"] for d in json.load(open(configs.val_path))]
    answers_val = [
        d["answer"].replace(",", "").strip() for d in json.load(open(configs.val_path))
    ]
    cot_val = ["\n".join(d["steps"]) for d in json.load(open(configs.val_path))]

    base_dataset_valid = get_dataset(
        configs.val_path,
        tokenizer,
        max_size=32 if configs.debug else 100000000,
        use_kge=configs.use_kge,
        kge_entity_to_id_path=kge_entity_to_id_path,
        kge_anchor_policy=configs.kge_anchor_policy,
    )

    if not configs.only_eval:
        base_dataset_train = get_dataset(
            configs.train_path,
            tokenizer,
            max_size=5000 if configs.debug else 100000000,
            use_kge=configs.use_kge,
            kge_entity_to_id_path=kge_entity_to_id_path,
            kge_anchor_policy=configs.kge_anchor_policy,
        )

    if "gsm" in configs.val_path:
        max_new_tokens = 64
    else:
        max_new_tokens = 128

    total_train_steps = 0

    if not configs.debug and not configs.only_eval and rank == 0:
        wandb_run = wandb.init(project=configs.project, name=configs.name, id=configs.id if configs.id else None, resume="allow")
        wandb_run.config.update(configs, allow_val_change=True)
        wandb_run.define_metric("train/step")
        wandb_run.define_metric("train/*", step_metric="train/step")
        text_table = wandb.Table(columns=["step", "text"])

    else:
        wandb_run = None

    optimizer = None
    scheduler = None
    optimizer_stage = None
    optimizer_group_names = []
    optimizer_group_parameters = {}

    best_acc = 0

    collator = MyCollator(
        tokenizer,
        latent_id=latent_id,
        label_pad_token_id=-100,
        use_kge=configs.use_kge,
    )

    for epoch in range(configs.resume, configs.num_epochs):
        scheduled_stage = (
            0 if (configs.cot or configs.no_cot) else epoch // configs.epochs_per_stage
        )
        dataset_gen_val = get_question_latent_dataset(
            scheduled_stage,
            base_dataset_valid,
            configs,
            start_id,
            latent_id,
            end_id,
            no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
        )

        valid_gen_dataloader = torch.utils.data.DataLoader(
            dataset_gen_val,
            num_workers=1,
            pin_memory=use_pin_memory,
            batch_size=1,
            collate_fn=collator,
            sampler=DistributedSampler(dataset_gen_val, shuffle=False),
        )

        if not configs.only_eval:
            dataset_train = get_cot_latent_dataset(
                scheduled_stage,
                base_dataset_train,
                configs,
                start_id,
                latent_id,
                end_id,
                no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
                shuffle=True,
            )

            train_dataloader = torch.utils.data.DataLoader(
                dataset_train,
                num_workers=1,
                shuffle=False,
                pin_memory=use_pin_memory,
                batch_size=configs.batch_size_training,
                collate_fn=collator,
                sampler=DistributedSampler(dataset_train, shuffle=True),
            )

            dataset_loss_val = get_cot_latent_dataset(
                scheduled_stage,
                base_dataset_valid,
                configs,
                start_id,
                latent_id,
                end_id,
                no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
            )

            valid_loss_dataloader = torch.utils.data.DataLoader(
                dataset_loss_val,
                num_workers=1,
                shuffle=False,
                pin_memory=use_pin_memory,
                batch_size=configs.batch_size_training,
                collate_fn=collator,
                sampler=DistributedSampler(dataset_loss_val, shuffle=False),
            )

            should_reset_optimizer = optimizer is None or (
                configs.reset_optimizer and optimizer_stage != scheduled_stage
            )
            if should_reset_optimizer:
                optimizer_groups, optimizer_named_params, missing_overrides = _build_optimizer_param_groups(
                    parallel_model,
                    lr=configs.lr,
                    lr_base_llm=getattr(configs, "lr_base_llm", None),
                    lr_projection_mlp=getattr(configs, "lr_projection_mlp", None),
                )
                optimizer = optim.AdamW(
                    [
                        {"params": group["params"], "lr": group["lr"]}
                        for group in optimizer_groups
                    ],
                    weight_decay=configs.weight_decay,
                )
                optimizer_stage = scheduled_stage
                scheduler = None
                optimizer_group_names = [group["name"] for group in optimizer_groups]
                optimizer_group_parameters = {
                    group_name: [param for _, param in named_params]
                    for group_name, named_params in optimizer_named_params.items()
                }
                expected_missing = set()
                if bool(getattr(configs, "freeze_base_llm", False)):
                    expected_missing.add("base_llm")
                if not bool(getattr(configs, "use_kge", False)):
                    expected_missing.add("projection_mlp")
                actionable_missing = [
                    group_name
                    for group_name in missing_overrides
                    if group_name not in expected_missing
                ]
                if rank == 0 and len(actionable_missing) > 0:
                    print(
                        "Warning: split LR override requested but no matching trainable parameters for groups:",
                        actionable_missing,
                    )

                scheduler_name = str(getattr(configs, "lr_scheduler", "none")).lower()
                if scheduler_name not in {"none", "cosine"}:
                    raise ValueError(
                        f"Unsupported lr_scheduler='{configs.lr_scheduler}'. Expected 'none' or 'cosine'."
                    )
                if scheduler_name == "cosine":
                    updates_per_epoch, total_scheduler_steps, warmup_steps = _scheduler_step_counts(
                        num_batches=len(train_dataloader),
                        gradient_accumulation_steps=configs.gradient_accumulation_steps,
                        epoch=epoch,
                        num_epochs=configs.num_epochs,
                        reset_optimizer=bool(configs.reset_optimizer),
                        epochs_per_stage=configs.epochs_per_stage,
                        single_stage_schedule=bool(configs.cot or configs.no_cot),
                        lr_warmup_ratio=configs.lr_warmup_ratio,
                    )
                    if total_scheduler_steps > 0:
                        scheduler = get_cosine_schedule_with_warmup(
                            optimizer,
                            num_warmup_steps=warmup_steps,
                            num_training_steps=total_scheduler_steps,
                        )
                    if rank == 0:
                        print(
                            "Scheduler initialized:",
                            {
                                "type": "cosine",
                                "epoch": epoch,
                                "stage": scheduled_stage,
                                "updates_per_epoch": updates_per_epoch,
                                "total_steps": total_scheduler_steps,
                                "warmup_steps": warmup_steps,
                            },
                        )

            parallel_model.module.train()

            total_length = max(len(train_dataloader), 1)
            pbar = tqdm(
                colour="blue",
                desc=f"Training Epoch: {epoch+1}",
                total=total_length,
                dynamic_ncols=True,
                disable=rank != 0,
            )
            optimizer_update_idx = 0

            for step, batch in enumerate(train_dataloader):
                if step == 0 and wandb_run and rank == 0:
                    if bool(getattr(configs, "wandb_log_training_data", True)):
                        print("logging training data")
                        cur_bs = len(batch["input_ids"])
                        max_examples = max(
                            int(getattr(configs, "wandb_training_data_max_examples", 2)),
                            0,
                        )
                        max_tokens_per_example = max(
                            int(
                                getattr(
                                    configs,
                                    "wandb_training_data_max_tokens_per_example",
                                    256,
                                )
                            ),
                            0,
                        )
                        max_chars = max(
                            int(getattr(configs, "wandb_training_data_max_chars", 20000)),
                            256,
                        )
                        text_lines = []
                        char_count = 0
                        truncated = False
                        for data_idx in range(min(cur_bs, max_examples)):
                            token_cap = min(
                                len(batch["input_ids"][data_idx]), max_tokens_per_example
                            )
                            for token_idx in range(token_cap):
                                line = (
                                    str(batch["input_ids"][data_idx][token_idx].item())
                                    + " "
                                    + str(batch["labels"][data_idx][token_idx].item())
                                    + " "
                                    + tokenizer.decode(batch["input_ids"][data_idx][token_idx])
                                )
                                line_len = len(line) + 1
                                if char_count + line_len > max_chars:
                                    truncated = True
                                    break
                                text_lines.append(line)
                                char_count += line_len
                            if truncated:
                                break
                            separator = "====" * 10
                            if char_count + len(separator) + 1 > max_chars:
                                truncated = True
                                break
                            text_lines.append(separator)
                            char_count += len(separator) + 1
                        if truncated:
                            text_lines.append("[truncated]")
                        text_table.add_data(total_train_steps, "\n".join(text_lines))
                        wandb_run.log({"data_table": text_table})

                total_train_steps += 1
                batch = {
                    key: batch[key].to(device) for key in batch.keys() if key != "idx"
                }

                outputs = parallel_model(**batch)

                loss = outputs.loss / configs.gradient_accumulation_steps
                loss.backward()

                grad_norm_to_log = None
                group_grad_norms_to_log = {}
                group_param_norms_to_log = {}
                if (step + 1) % configs.gradient_accumulation_steps == 0 or step == len(
                    train_dataloader
                ) - 1:
                    max_grad_norm = getattr(configs, "max_grad_norm", None)
                    if max_grad_norm is not None and float(max_grad_norm) > 0:
                        if hasattr(parallel_model, "clip_grad_norm_"):
                            grad_norm = parallel_model.clip_grad_norm_(float(max_grad_norm))
                        else:
                            grad_norm = nn_utils.clip_grad_norm_(
                                _trainable_parameters(parallel_model), float(max_grad_norm)
                            )
                        if isinstance(grad_norm, torch.Tensor):
                            grad_norm_to_log = grad_norm.detach().float()
                        else:
                            grad_norm_to_log = torch.tensor(float(grad_norm), device=device)

                    norm_log_interval = max(
                        int(getattr(configs, "wandb_group_norm_log_interval", 50)), 1
                    )
                    should_log_group_norms = (
                        optimizer_update_idx % norm_log_interval == 0
                    )
                    if should_log_group_norms:
                        if bool(getattr(configs, "wandb_log_group_grad_norm", False)):
                            for group_name in optimizer_group_names:
                                group_parameters = optimizer_group_parameters.get(group_name, [])
                                group_grad_norm = _group_grad_norm(group_parameters, device=device)
                                if group_grad_norm is not None:
                                    group_grad_norms_to_log[group_name] = group_grad_norm
                        if bool(getattr(configs, "wandb_log_group_param_norm", False)):
                            for group_name in optimizer_group_names:
                                group_parameters = optimizer_group_parameters.get(group_name, [])
                                group_param_norms_to_log[group_name] = _group_param_norm(
                                    group_parameters, device=device
                                )

                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()
                    optimizer_update_idx += 1

                if wandb_run and rank == 0:
                    log_dict = {
                        "train/epoch": epoch + 1,
                        "train/step": epoch * len(train_dataloader) + step,
                        "train/loss": loss.detach().float()
                        * configs.gradient_accumulation_steps,
                    }
                    if getattr(outputs, "align_loss", None) is not None:
                        log_dict["train/align_loss"] = outputs.align_loss.detach().float()
                    if getattr(outputs, "kge_embedding", None) is not None:
                        log_dict["train/kge_embed_norm"] = (
                            outputs.kge_embedding.detach().float().norm(dim=-1).mean()
                        )
                    if getattr(outputs, "anchor_coverage", None) is not None:
                        log_dict["train/anchor_coverage"] = outputs.anchor_coverage.detach().float()
                    if optimizer is not None:
                        lr_by_group = {}
                        for group_name, group in zip(optimizer_group_names, optimizer.param_groups):
                            group_lr = group["lr"]
                            lr_by_group[group_name] = group_lr
                            log_dict[f"train/lr/{group_name}"] = group_lr
                        if "base_llm" in lr_by_group:
                            log_dict["train/lr"] = lr_by_group["base_llm"]
                        else:
                            log_dict["train/lr"] = optimizer.param_groups[0]["lr"]
                    if grad_norm_to_log is not None:
                        log_dict["train/grad_norm"] = grad_norm_to_log
                    for group_name, grad_norm in group_grad_norms_to_log.items():
                        log_dict[f"train/grad_norm/{group_name}"] = grad_norm
                    for group_name, param_norm in group_param_norms_to_log.items():
                        log_dict[f"train/param_norm/{group_name}"] = param_norm
                    wandb_run.log(log_dict)
                pbar.update(1)

                pbar.set_description(
                    f"Training Epoch: {epoch+1}/{configs.num_epochs}, batch {step}/{len(train_dataloader)} "
                    f"completed (loss: {round(float(loss.detach().float() * configs.gradient_accumulation_steps), 4)}"
                )
            pbar.close()
            dist.barrier()

            if (
                not configs.save_only_improve
                and not configs.debug
                and not configs.only_eval
            ):
                states = parallel_model.state_dict()
                if rank == 0:
                    torch.save(states, os.path.join(save_dir, f"checkpoint_{epoch + 1}"))
                    print("saving model.")

                dist.barrier()
                del states
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            total_loss = 0

            with torch.no_grad():
                parallel_model.module.eval()
                for step, batch in enumerate(valid_loss_dataloader):
                    batch = {
                        key: batch[key].to(device) for key in batch.keys() if key != "idx"
                    }

                    outputs = parallel_model(**batch)
                    loss = outputs.loss
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    total_loss += loss.item() / world_size

                if wandb_run and rank == 0:
                    log_dict = {
                        "eval/loss": total_loss / len(valid_loss_dataloader),
                    }
                    wandb_run.log(log_dict)
                    print("eval loss", total_loss / len(valid_loss_dataloader))

        total_length = len(valid_gen_dataloader)

        pbar = tqdm(
            colour="blue", desc="Test Accuracy", total=total_length, dynamic_ncols=True
        )
        cor, cor_cot, total = (
            torch.tensor(0, device=device),
            torch.tensor(0, device=device),
            torch.tensor(0, device=device),
        )
        generated_tokens_sum = torch.tensor(0.0, device=device)

        with torch.no_grad():
            parallel_model.module.eval()
            for idx, batch in enumerate(valid_gen_dataloader):
                test_idx = batch["idx"][0]

                batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if v is not None and k not in ["idx", "position_ids"]
                }

                assert len(batch["input_ids"]) == 1
                answer = answers_val[test_idx.cpu().item()]
                answer_cot = cot_val[test_idx.cpu().item()]
                question = question_val[test_idx.cpu().item()]

                total += 1

                gen_ctx = (
                    FSDP.summon_full_params(parallel_model, writeback=False, recurse=True)
                    if torch.cuda.is_available() and not configs.only_eval
                    else nullcontext()
                )
                with gen_ctx:
                    outputs = parallel_model.module.generate(
                        **batch,
                        max_new_tokens=max_new_tokens,
                        synced_gpus=not configs.only_eval,
                    )

                text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer_output = text_output.split("#")[-1].replace(",", "").strip()
                cot_output = (("\n".join(text_output.split("\n")[1:])).split("#")[0].strip())
                generated_tokens_sum += outputs.shape[1] - batch["input_ids"].shape[1]

                if idx < 5 and rank == 0:
                    print(
                        f"Question {test_idx}: Answer = '{answer}' CoT = '{answer_cot}'"
                    )
                    print(f"Full output: '{tokenizer.decode(outputs[0])}'")
                    print(f"Extracted Output: '{answer_output}'")

                cor += answer_output == answer
                cor_cot += cot_output == answer_cot

                pbar.update(1)
                pbar.set_description(
                    f"Test accuracy: {round(float(cor.detach().float() / total.detach().float()), 2)}"
                )

            pbar.close()
            print(f"Device {rank}: Cor={cor}, CoT={cor_cot}, Total={total}")

        dist.all_reduce(cor_cot, op=dist.ReduceOp.SUM)
        dist.all_reduce(cor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        dist.all_reduce(generated_tokens_sum, op=dist.ReduceOp.SUM)

        cor_cot = cor_cot.item()
        cor = cor.item()
        total = total.item()
        generated_tokens_total = generated_tokens_sum.item()
        avg_generated_tokens = generated_tokens_total / max(total, 1)
        if rank == 0:
            print(f"Average generated tokens: {avg_generated_tokens}")
            print(f"Accuracy on validation set: {cor} / {total} = {cor/total}")
            print(f"CoT match on validation set: {cor_cot} / {total} = {cor_cot/total}")
        sys.stdout.flush()

        if wandb_run:
            wandb_run.log(
                {
                    "eval/acc": cor / total,
                    "eval/cot_em": cor_cot / total,
                    "eval/generated_tokens_avg": avg_generated_tokens,
                }
            )

        if configs.only_eval:
            break

        dist.barrier()
        if (
            cor / total > best_acc
            and configs.save_only_improve
            and not configs.debug
            and not configs.only_eval
        ):
            states = parallel_model.state_dict()

            if rank == 0:
                torch.save(states, os.path.join(save_dir, f"checkpoint_{epoch + 1}"))
                print("saving model.")

            best_acc = cor / total

            dist.barrier()
            del states
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
