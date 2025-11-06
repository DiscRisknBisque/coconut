# GNN Encoder Integration Tasks

1. **Dependencies**
   - Add PyTorch Geometric (and required scatter packages) to `requirements.txt`, matching the repo’s Torch/CUDA matrix.

2. **Graph Encoder Module**
   - Create `graph_encoder.py` with the 2-layer GCN `GraphEncoder` class, projection MLP `P_g`, and a registry to instantiate encoders from YAML (`graph_encoder: "gcn2"`).

3. **ProntoQA Graph Preprocessing**
   - Implement `preprocessing/graphify_prontoqa.py` to build per-example graphs, compute hashed node features + role one-hot, add lexical and sequential edges, and serialize sidecars under `data/prontoqa_graphs/{split}/{idx}.pt` with manifest if needed.

4. **ProsQA Graph Preprocessing**
   - Implement `preprocessing/graphify_prosqa.py` mirroring the ProntoQA script but sourcing propositions from ProsQA JSON; ensure consistent caching layout.

5. **Dataset Integration**
   - Update `dataset.py` to locate graph sidecars for each item, load tensors in `__getitem__`, and extend the collate function to construct PyG-compatible `batch`, `edge_index`, `x`, `role`, and attach them to the model batch when `use_graph` is enabled.

6. **Model Conditioning**
   - Modify `coconut.py` to instantiate the graph encoder & projection when configured, derive `Z_g`, inject graph-conditioned soft prompts (length = `graph_prefix_len`) into the token embeddings, apply residual latent nudges at the start of each latent segment, and add the cosine alignment loss (`align_loss_weight`).

7. **Configuration Files**
   - Add new YAML configs `args/prontoqa_coconut_gnn.yaml` and `args/prosqa_coconut_gnn.yaml` that clone the originals and append graph-specific flags (`use_graph`, `graph_encoder`, `graph_dim`, `graph_prefix_len`, `align_loss_weight`, `latent_injection`).

8. **Training & Logging Adjustments**
   - Ensure training scripts load the new configs, log alignment loss and cosine metrics, and report latent-step efficiency comparisons between baseline and GNN runs.

9. **Unit & Regression Tests**
   - Add quick tests covering graph encoder output shapes, `.pt` sidecar save/load, and regression checks confirming logits match baseline when `use_graph=false` and that `align_loss_weight=0` disables the extra loss term.

