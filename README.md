# Coconut

The code base is the official implementation of [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769).

![coconut](assets/coconut.png)

## Getting Started

Clone repo:

```
git clone git@github.com:facebookresearch/coconut.git
cd coconut
```

Setup environment:

```
conda create --name coconut python=3.12
conda activate coconut
pip install -r requirements.txt
```

> **Note:** RotatE training uses [PyKEEN](https://github.com/pykeen/pykeen). Use Python 3.11+ and install dependencies from `requirements.txt` before running the offline KGE scripts.

The code relies on [wandb](https://wandb.ai/site/) for logging. Please log in your wandb account following this [document](https://docs.wandb.ai/ref/cli/wandb-login/) before running any experiments.

## Data

The data for training and evaluation should be presented as a json file like below:

```python
[
  {
    "question": "...",
    "answer": "...",
    "steps": ["...", "...", ...]
  },
  ...
]
```

The file should contain a list of data points. Each data point is composed of a question (str), an answer (str), and a list of steps (str), where each of them is a string.

For example, you can download and process the [GSM8K](https://arxiv.org/abs/2110.14168) dataset (with [augmented training and validation sets](https://github.com/da03/Internalize_CoT_Step_by_Step/tree/e06a32ee5e4cd117171daeb4755d2a97ece62761/data/gsm8k)) by running:

```bash
bash preprocessing/gsm_icot.bash
```

## RotatE KGE Pipeline (ProsQA)

This repo uses an offline PyKEEN RotatE pipeline for graph conditioning. Export triples first, then train RotatE, then train Coconut with the KGE projector config.

```bash
python preprocessing/export_prosqa_triples.py \
  --input-dir data \
  --output-dir data/prosqa_rotate

python preprocessing/train_rotate_pykeen.py \
  --triples-dir data/prosqa_rotate \
  --output-dir data/prosqa_rotate \
  --embedding-dim 256
```

Artifacts expected by runtime are written under `data/prosqa_rotate/`:

- `entity_embeddings.pt`
- `relation_embeddings.pt`
- `entity_to_id.json`
- `relation_to_id.json`
- `metadata.json`

## Latent Cartographer (ProsQA)

`analysis/latent_cartographer.py` decodes Coconut's latent trajectory into nearest projected KGE symbols and writes map/path artifacts for one ProsQA sample.

Install analysis dependencies:

```bash
pip install scikit-learn matplotlib pillow
```

Run cartography on one validation sample:

```bash
python analysis/latent_cartographer.py \
  --config args/prosqa_coconut_rotate.yaml \
  --checkpoint models/checkpoint_49 \
  --split test \
  --sample-idx 1 \
  --metric cosine \
  --output-dir analysis_outputs \
  --save-gif
```

Outputs are written under `analysis_outputs/<split>_<sample_idx>/`:

- `node_embeddings.pt`
- `reasoning_trajectory.pt`
- `similarity.pt`
- `decoded_path.json`
- `metrics.json`
- `cartography.png`
- `cartography.gif` (when `--save-gif` is set)

## Arguments

The configuration of a run should be specified in a yaml file (an example can be found [here](args/gsm_coconut.yaml)).

- **General settings**
  - **project**: Project name for wandb
  - **save_path**: Your path to store the checkpoints
  - **only_eval**: If true, only load a model and test on the data from `val_path` (must used along with `load_model_path`). Otherwise, train the model on `train_path` and test on `val_path` after every epoch.

- **Method**
  - **coconut**: Train coconut model
  - **cot**: Train cot model
  - **no_thoughts**: Train coconut (w/o thought) model
  - **no_cot**: Train no-cot model

- **Training settings**
  - **c_thought**: Number of continuous thoughts for each reasoning step
  - **epochs_per_stage**: Number of epochs for every training stage
  - **max_latent_stage**: The maximum number of training stages (in addition to the initial stage)
  - **pad_latent_to_max**: If the number of reasoning steps is fewer than the index of current training stage, pad the number of continuous thoughts.
  - **save_only_improve**: Save the model only when there the best validation accuracy is updated. Recommended to set `False` for Coconut model training, because otherwise the checkpoints in the last stage might now get saved.
  - **uniform_prob**: The probability to mix data from other stages. 0 for standard experiment, 0.3 for analysis experiment.
  - **model_id**: Huggingface model id to load as the initialization, e.g., `openai-community/gpt2`
  - **load_model_path**: The path to a checkpoint to load. Used in two cases: (1) for evaluation (2) to initialize coconut from a CoT-tuned model.
  - **seed**: Random seed.
  - **resume**: The epoch to resume. Can be used when we want to skip the initial training stages.
  - **bf16**: Whether to use bf16 training.
  - **train_path**: Path to the training set.
  - **val_path**: Path to the validation or test set (depending on `only_eval`)
  - **reset_optimizer**: Whether to reset the optimizer when swtiching training stages.
  - **batch_size_training**: Batch size to train the model per GPU.
  - **debug**: If true, there is no wandb and model saving. A subset of data will be used.
  - **gradient_accumulation_steps**: Gradient accumulation steps
  - **num_epochs**: Maximum training epoches.
  - **lr**: Learning rate
  - **lr_base_llm**: Optional override for base LLM learning rate (`base_causallm.*`). Falls back to `lr`.
  - **lr_projection_mlp**: Optional override for projector learning rate (`kge_projector.*` and `kge_residual_norm.*`). Falls back to `lr`.
  - **weight_decay**: Weight decay
  - **lr_scheduler**: Learning rate scheduler (`none`, `cosine`).
  - **lr_warmup_ratio**: Warmup fraction of optimizer-update steps (e.g. `0.1` for 10%).
  - **max_grad_norm**: Gradient clipping max norm. Disabled when unset/`None`/`<= 0`.

- **KGE settings**
  - **use_kge**: Enable RotatE-conditioned Coconut path.
  - **kge_artifact_root**: Directory containing exported KGE artifacts.
  - **kge_entity_embeddings_file**: Entity embedding tensor filename under `kge_artifact_root`.
  - **kge_entity_to_id_file**: Entity-id mapping filename under `kge_artifact_root`.
  - **kge_metadata_file**: Metadata filename with `projector_in_dim`.
  - **kge_anchor_policy**: Anchor strategy. Current implementation supports `query_anchors`.
  - **kge_projector_hidden**: Hidden size for the KGE projector MLP.
  - **kge_projector_num_hidden_layers**: Number of hidden layers in the KGE projector MLP (default `1`).
  - **kge_projector_activation**: Projector activation (`gelu`, `relu`, `none`).
  - **kge_projector_layernorm**: Whether to apply LayerNorm after projector output.
  - **latent_injection**: Latent residual policy (`residual`, `none`).
  - **align_loss_weight**: Weight of cosine alignment loss.
  - **freeze_base_llm**: Freeze base LLM weights and train only KGE-conditioning modules.

## Training

Run the following commands (replacing `N_GPUS` and `PATH_TO_ARGS`):

```
torchrun --nnodes 1 --nproc_per_node N_GPUS run.py PATH_TO_ARGS
```

When `lr_scheduler: cosine` is enabled, warmup and cosine decay are computed over optimizer-update steps (not raw dataloader batches), so `gradient_accumulation_steps` is handled correctly.

## Reproducing Experiments

Here we provide instructions to reproduce our experiments in the paper.

All the commands below assume 4 \* A100 (80GB) GPUs. You may change the corresponding arguments in the config file (`batch_size_training`, `gradient_accumulation_steps`) and `nproc_per_node` when launching the run, to adapt your resources.

### GSM8K

Preprocessing data:

```bash
bash preprocessing/gsm_icot.bash
```

First train the model with CoT (as the stage 0 training)

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_cot.yaml
```

Select a checkpoint as the initialization of Coconut (the validation accuracy is expected to be around 40%). Replace the `load_model_path` in the [args/gsm_coconut.yaml](args/gsm_coconut.yaml) with your selected checkpoint, and run:

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_coconut.yaml
```

Find the checkpoint with best validation accuracy, and put the path as `load_model_path` in [args/gsm_coconut_eval.yaml](args/gsm_coconut_eval.yaml). To evaluate:

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_coconut_eval.yaml
```

### ProntoQA

Please clone the official [github repo](https://github.com/asaparov/prontoqa/tree/f0145b867b3c106285ec9ea1941a3f6eb7c6162d) of [ProntoQA](https://arxiv.org/pdf/2210.01240) and generate a raw dataset with:

```bash
cd prontoqa
python run_experiment.py --model-name json --model-size dummy --ordering random --num-trials 10000 --few-shot-examples 0 --ontology fictional --min-hops 5 --max-hops 5 --hops-skip 1
```

Then copy the generated `5hop_0shot_random.json` file to `data` directory, and preprocess the dataset with:

```bash
python preprocessing/prontoqa.py
```

Then run the following to train the model:

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/prontoqa_coconut.yaml
```

ProntoQA currently uses the baseline Coconut config:

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/prontoqa_coconut.yaml
```

Find the checkpoint with best validation accuracy, and put the path as `load_model_path` in [args/prosqa_coconut_eval.yaml](args/prosqa_coconut_eval.yaml). To evaluate:

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/prosqa_coconut_eval.yaml
```

### ProsQA

The ProsQA dataset is at [data/prosqa\_\*.json](data).

Then run the following to train the model:

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/prosqa_coconut.yaml
```

Run the RotatE-conditioned variant with:

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/prosqa_coconut_rotate.yaml
```

Find the checkpoint with best validation accuracy, and put the path as `load_model_path` in [args/prosqa_coconut_eval.yaml](args/prosqa_coconut_eval.yaml). To evaluate:

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/prosqa_coconut_eval.yaml
```

## Citation

If you use this code base in your research, please cite our paper with the following BibTex entry:

```bibtex
@article{hao2024training,
  title={Training Large Language Models to Reason in a Continuous Latent Space},
  author={Hao, Shibo and Sukhbaatar, Sainbayar and Su, DiJia and Li, Xian and Hu, Zhiting and Weston, Jason and Tian, Yuandong},
  journal={arXiv preprint arXiv:2412.06769},
  year={2024}
}
```

## License

This code is released under the MIT license (see [LICENSE](LICENSE)).
