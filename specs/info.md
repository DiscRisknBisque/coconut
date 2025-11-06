Here’s a focused spec your coding agent can follow to implement the **GNN subgraph encoder** and wire it into **Coconut** (GPT-2 base) for **ProntoQA** and **ProsQA**.

---

# Objective

Add a _minimal_ graph signal to Coconut by conditioning its latent loop on a compact **subgraph embedding** `Z_g`. No tokenizer changes, no architectural churn: just a tiny GNN, soft-prompt injection, and one small alignment loss.

**Success signal** (vs. Coconut baseline, same compute): equal or higher EM on ProntoQA/ProsQA and/or fewer latent steps at equal EM.

---

# Repos, files, and where to patch

- **Base repo**: fork `facebookresearch/coconut` and work in a branch `gnn-encoder` (already complete). The repo provides GPT-2 loading, staged Coconut training, and ProntoQA/ProsQA pipelines. ([GitHub][1])
- **Add files**

  - `graph_encoder.py` — the tiny GNN and a registry.
  - `preprocessing/graphify_prontoqa.py` — build per-example subgraphs for ProntoQA.
  - `preprocessing/graphify_prosqa.py` — build per-example subgraphs for ProsQA.

- **Modify files**

  - `dataset.py` — load graph sidecars; add to batch.
  - `coconut.py` — inject soft prompts, add residual latent injection, compute alignment loss.
  - `requirements.txt` — add PyTorch Geometric (see install note). ([PyTorch Geometric][2])

---

# Data assumptions (Coconut format & datasets)

Coconut expects JSON with `{"question": str, "answer": str, "steps": [str, ...]}` and includes **commands for ProntoQA** (generate JSON with the official repo) and **ProsQA** (JSONs under `data/prosqa_*.json`). We keep that intact and add **sidecar** graph tensors. ([GitHub][1])

- **ProntoQA**: generate `5hop_0shot_random.json` with `asaparov/prontoqa`, then `python preprocessing/prontoqa.py` as Coconut instructs. ([GitHub][1])
- **ProsQA**: use the JSONs in `coconut/data/prosqa_*.json`. ([GitHub][1])

---

# Subgraph construction (fast, robust, parser-free)

**Goal**: create a small, always-available graph per datum without brittle FOL parsing.

For each example:

1. **Nodes**

   - One node per **context sentence** we pass to the model. Use:

     - ProntoQA: all **premise/context lines** and (optionally) each **step**.
     - ProsQA: each **premise/proposition** string present in the item’s JSON.

   - Optionally add a single **[QUESTION]** node with the question text (helps ProsQA).

2. **Edges (undirected)**

   - Add edge `(i, j)` if sentences share ≥1 non-stopword lemma (simple lexical overlap).
   - Add **sequential edges** along the given `steps` list: step*k ↔ step*{k+1}.
   - (Optional later) mark derived-from edges if a step string appears in both premise and steps.

3. **Node features (minimal)**

   - `x_i = [bow_hash(uni-/bi-grams, dim=256)] ⊕ [role_one_hot( premise/step/question )]`
   - Hashing trick: fixed random projection from token ids to 256 dims; normalize.

4. **Pooling**

   - We’ll compute node embeddings with the GNN and **mean-pool** to `Z_g ∈ ℝ^{d_g}` (default `d_g=256`).

**Caching**
Write a sidecar per example: `graphs/{split}/{index}.pt` with `{"edge_index": LongTensor[2,E], "x": FloatTensor[N,256], "role": LongTensor[N]}`. Store the path on the dataset item id.

---

# GNN encoder (tiny, PyTorch Geometric)

**Install** PyG matching your Torch/CUDA (follow official matrix to avoid ABI mismatches). ([PyTorch Geometric][2])

**Model** `GraphEncoder(d_in=256+3, d_hidden=256, d_out=256)`:

```python
# graph_encoder.py
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GraphEncoder(nn.Module):
    def __init__(self, d_in=259, d_hidden=256, d_out=256, dropout=0.1):
        super().__init__()
        self.conv1 = GCNConv(d_in, d_hidden, cached=False, add_self_loops=True)
        self.conv2 = GCNConv(d_hidden, d_out, cached=False, add_self_loops=True)
        self.ln1 = nn.LayerNorm(d_hidden)
        self.ln2 = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index); h = self.ln1(F.relu(h)); h = self.dropout(h)
        h = self.conv2(h, edge_index); h = self.ln2(h)
        Z_g = global_mean_pool(h, batch)  # [B, d_out]
        return Z_g, h  # (global, per-node)
```

> `batch` is the standard PyG vector mapping nodes→graph index; since we process 1 example at a time in preprocessing, `batch` will be created in the collate step.

**Registry**
Add a small factory so YAML can set `graph_encoder: "gcn2"`.

---

# Integration with Coconut (minimal conditioning)

We will **not** touch tokenization. We feed graph information by:

1. **Soft-prompt injection** (prefix embeddings)

   - Project `Z_g` to LLM hidden size `d_ℓ` with a tiny MLP `P_g`.
   - Create `graph_prefix_len = 4` **virtual tokens** as learned vectors initialized from `P_g(Z_g)` via FiLM-like MLP.
   - **Prepend** these vectors to the input embedding sequence for each example.

2. **Latent-loop residual nudge**

   - At the _start_ of each `<bot> … <eot>` latent segment, **add** `P_g(Z_g)` (LayerNorm + residual scaling) to the first latent state before it’s fed back as the “continuous thought”.

3. **Alignment loss (single cosine term)**

   - Encourage the model’s pre-decode hidden state to align with `P_g(Z_g)`:
     [
     \mathcal{L}*{align} = 1 - \cos\big( \text{normalize}(P_g(Z_g)),\ \text{normalize}(h*{\text{pre-decode}}) \big)
     ]
   - Weight with `align_loss_weight = 0.05`.

**Code touchpoints**

- `dataset.py`

  - Extend the item to return `graph_path` or tensors.
  - In collate, load `edge_index, x, role`; build PyG `batch` (all zeros if one graph per sample); put in the batch dict.

- `coconut.py`

  - Instantiate `GraphEncoder` and `P_g` when `use_graph: true`.
  - Forward pass:

    - Compute `Z_g, h_nodes = graph_encoder(x, edge_index, batch)`.
    - Build soft-prompt embeddings from `P_g(Z_g)` and **prepend** to input embeddings.
    - On entering each **latent** segment, add the residual `P_g(Z_g)` to the first latent hidden state.
    - Compute `L_align` using the hidden state just before emitting the next _visible_ token after the latent loop; add to total loss.

- `args/*.yaml` (new)

  - `use_graph: true`
  - `graph_encoder: "gcn2"`
  - `graph_dim: 256`
  - `graph_prefix_len: 4`
  - `align_loss_weight: 0.05`
  - `latent_injection: "residual"`

(Everything else stays exactly as in Coconut’s ProntoQA/ProsQA YAMLs.) ([GitHub][1])

---

# Preprocessing scripts (what the agent should implement)

## `preprocessing/graphify_prontoqa.py`

Input: the JSON produced by Coconut’s `preprocessing/prontoqa.py`. Output: a mirrored file tree `data/prontoqa_graphs/{split}/{idx}.pt`.

Steps per item:

1. **Collect sentences**

   - From the preprocessed item, extract context/premises. If steps are present, optionally include them as nodes of `role=step`. Add the question as `role=question` (single node) if you like.

2. **Tokenize & normalize**

   - Lowercase, strip punctuation; remove stopwords; lemmatize (simple WordNet or porter stemmer).

3. **Edges**

   - For each pair `(i,j)`, add edge if `|tokens_i ∩ tokens_j| ≥ 1`.
   - Add edges `(step_k, step_{k+1})`.

4. **Features**

   - Build 256-dim hashing trick vector for each sentence.
   - Append a 3-dim one-hot role embedding.

5. **Serialize**

   - Save tensors to `.pt` keyed by example index. Persist maps in a manifest JSON if helpful.

## `preprocessing/graphify_prosqa.py`

Same recipe; sources for nodes are the proposition/premise strings inside ProsQA JSON (under `coconut/data/prosqa_*.json`). ([GitHub][1])

---

# Training & eval flow (unchanged Coconut + our flags)

Coconut README gives exact commands for ProntoQA/ProsQA generation, training, and evaluation (stage-0 CoT → Coconut). Keep those commands; just point the Coconut stages to our GNN YAMLs. ([GitHub][1])

**New YAMLs** (copy originals, then add our keys):

- `args/prontoqa_coconut_gnn.yaml`
- `args/prosqa_coconut_gnn.yaml`

**Sanity runs**

- **Baseline**: `args/prontoqa_coconut.yaml`, `args/prosqa_coconut.yaml`
- **Ours**: `args/*_coconut_gnn.yaml`
- **Ablations**: set `use_graph: false`, `align_loss_weight: 0`, or `latent_injection: none`.

---

# Defaults & hyperparameters

- `graph_dim = 256`, `graph_prefix_len = 4`, `align_loss_weight = 0.05`
- GNN: 2-layer GCN, dropout 0.1, LayerNorm on both layers.
- Hashing trick: 50k unigram vocab → 256-dim signed hash with √N scaling.
- Injection: residual scale 1.0 with pre-LayerNorm; clamp ‖P_g(Z_g)‖ via RMSNorm if training becomes unstable.
- Leave Coconut’s `c_thought`, `max_latent_stage`, LR, and batch sizes as in the repo’s ProntoQA/ProsQA configs.

---

# Metrics to log

- **Primary**: EM accuracy (Coconut already reports).
- **Efficiency**: average latent steps per sample; wall-clock/sample.
- **Ablations**: deltas for (no-graph / align-only / inject-only / full).
- **Stability**: running cosine similarity between `P_g(Z_g)` and `h_pre-decode`.

---

# Unit tests (quick)

- **Shape test**: `GraphEncoder` returns `Z_g: [B,256]`, `h_nodes: [N,256]`.
- **Serialization**: round-trip save/load of `.pt` sidecars.
- **Injection**: with `use_graph=false`, model logits must match original Coconut within 1e-6 on a fixed seed batch.
- **Loss on/off**: setting `align_loss_weight=0` zeroes the additional loss term.

---

# Troubleshooting

- **PyG install errors (ABI/CUDA)**: ensure Torch + PyG wheels match (the PyG docs explain the version matrix and common CUDA mismatch errors). ([PyTorch Geometric][2])
- **Exploding `L_align`**: normalize both vectors; reduce `align_loss_weight` to 0.01; add gradient clipping 1.0.
- **No accuracy change**: try `graph_prefix_len=2/8`; swap GCN→GraphSAGE; include the `[QUESTION]` node; or disable sequential edges.

---

# Why this is minimal

- Uses **Coconut as-is** (GPT-2 base, staged training, same eval scripts and data format). ([GitHub][1])
- Adds a **single** small module + **one** extra loss term.
- No token, vocab, or YAML upheaval beyond a few new flags.
- Graph build is **parser-free** and deterministic across both benchmarks.

---

## References for the agent

- **Coconut paper & repo** (latent loop; ProntoQA/ProsQA commands; GPT-2 load). ([arXiv][3])
- **ProntoQA repo** (JSON generation flags and format). ([GitHub][4])
- **PyTorch Geometric install notes** (version/CUDA matching). ([PyTorch Geometric][2])

If you want, I can also draft the two new `graphify_*.py` scripts and a `GraphEncoder` stub matching these shapes so your agent can start from exact scaffolding.

[1]: https://github.com/facebookresearch/coconut "GitHub - facebookresearch/coconut: Training Large Language Model to Reason in a Continuous Latent Space"
[2]: https://pytorch-geometric.readthedocs.io/en/2.4.0/install/installation.html?utm_source=chatgpt.com "Installation — pytorch_geometric documentation"
[3]: https://arxiv.org/pdf/2412.06769?utm_source=chatgpt.com "arXiv:2412.06769v1 [cs.CL] 9 Dec 2024"
[4]: https://github.com/asaparov/prontoqa "GitHub - asaparov/prontoqa: Synthetic question-answering dataset to formally analyze the chain-of-thought output of large language models on a reasoning task."
