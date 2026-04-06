# Lattice3 

What it is: An end-to-end tri-modal deep learning pipeline for protein fold classification, integrating: ViT, BERT and GPSConv (Graph Transformer), with MLOps-centric design (Advanced ETL pipelines and scalability-friendly design) and Dockerized reproducibility.

What it does: Predicts protein structure classifications, for CATH domain proteins.

###  Architecture: -
<img width="842" height="240" alt="image" src="https://github.com/user-attachments/assets/459862f1-1248-456c-9224-7a337e555404" />

```
How it works: -
  1. The preprocessing script genrates the required data.
  2. ViT is finetuned with the dense contact maps, preferraly by unfreezing last 2 or 3 blocks.
  3. ViT is now used to generate dense embeddings from contact maps.
  4. ProtBERT is used to generate dense embeddings from FASTA sequences per residue (NOT POOLED).
  5. Graph Transformer block (TransformerConv) takes in ProtBERT embeddings as node_features, sparse contact map as edge_index,
     and the labels, and produces a 320 dim embedding.
  6. ViT and TFConv's embeddings are cross queried (Graph=Q,ViT=K/V) and the final output is logits from FFN.
```

## MLOps Design

Lattice3 is built as a **stage-decoupled system with materialized representations**, not a monolithic end-to-end pipeline.

- **Modular execution**  
    Stages (Preprocess → ViT → ProtBERT → GraphTF) run independently with persisted `.pt` outputs. Expensive embeddings are computed once and reused.

- **Representation-first workflow**  
    Samples are incrementally enriched (`dense_map → esm_residue_emb → vit_struct_emb`), enabling partial recomputation without full pipeline resets.

- **Stage-aligned tracking**  
    MLflow experiments are split per component, allowing clear attribution of performance gains.

- **Profiled inference DAG**  
    Latency is logged per stage (`preprocess / vit / protbert / graph`), exposing true bottlenecks.

- **Failure isolation**  
    Errors are tagged by stage, avoiding end-to-end debugging ambiguity.

- **Config-driven reproducibility**  
    All stages are governed by config (`load_cfg`) for consistent reruns.

- **Bounded data construction**  
    Graph inputs are constrained (`MAX_RESIDUES`, `DIST_THRESHOLD`, `EDGE_CAP`) to stabilize input distributions.

> Result: ~450M parameter system, operationally lightweight—modular, cache-efficient, and easy to debug.
### Structure Dynamics: -

```
The reason I engineered this tri-modal architectue: -
  1. ViT encodes global spatial geometry, derived from 2D spatial relationships. This gives us an idea of how the overall protein
     itself is connected on a higher level
  2. BERT encodes biochemical and evolutionary information mainly. This gives an idea of what the residues actually are.
  3. Graph Transformer (GPSConv) encodes structural topology and interaction contexts. It understands how residues are connected, which
     ones interact, and how structural neighborhoods are organized within the protein.
  4. Graph Transformer's embeddings queries the ViT embedding to produce a fused structural representation combining topology and geometry.
  5. The result is a latent space where structure, sematics and topology are unified and represented.
```

### Results: -

1. Accuracy: -
   <img width="1338" height="368" alt="image" src="https://github.com/user-attachments/assets/b7c47202-1d90-42a3-a0f2-f3d71065d65e" />
2. Macro F1: -
   <img width="1341" height="368" alt="image" src="https://github.com/user-attachments/assets/6eadcc02-cdaf-4eef-bf18-eeab4bb19912" />

| Model        | Test Accuracy (%) | Macro F1 |
|--------------|------------------|----------|
| ViT          | 83.5             | 0.82     |
| ProtBERT     | 84.2             | 0.83     |
| Fusion (ViT + GPSConv) | 89.0             | 0.88     |
3. a. Predicted accuracy and macro-F1 from ViT model alone shows that global reasoning contains class information to an extent.\
    b. Same inference from BERT model too, as protein chemistry and local interactivity carries class information too. \
    c. The fusion model outperforms by alomst 5-6%, as it provides a latent space where local and global trends are represented unified, hence the accuracy boost.

4. FastAPI pipeline: -
   <img width="1919" height="1138" alt="Screenshot 2026-03-24 113738" src="https://github.com/user-attachments/assets/fee521a6-cfb1-48c6-acd7-3e9310db5b4e" />





### Misc informations: -

| Component | Configuration | Purpose |
| :--- | :--- | :--- |
| **(BERT)Sequence Encoder** | HuggingFace `ProtBERT` ( FP16, Max Len: 1024 ) | Acts as a frozen language model to extract per-residue features from raw FASTA sequences. |
| **(ViT)Structural Encoder** | timm `vit_small_patch16_224` | Processes dense contact maps to generate a global 320-dimensional structural embedding. |
| **(TransformerConv) Graph Transformer** | PyG `GPSConv` ( 2 Layers ) + `SAGEConv` | Captures topological layout using sparse maps (edges) and ProtBERT embeddings (nodes) to produce a 320-dim graph embedding. |
| **Multimodal Fusion** | `nn.MultiheadAttention` ( dim=320, heads=4 ) | Cross-queries the topological graph representation (Query) against the structural ViT embedding (Key/Value). |
| **Classifier output** | Linear FFN ( dropout=0.118, out_features=4 ) | Maps the fused 320-dimensional multimodal representation to the final 4-class protein fold logits. |

1. System used: i7-13650HX, RTX4060 8GB GPU
2. Execution time (whole pipeline excluding downloads): 25-34 mins; 2-5s (single protein)  
3. Memory footprint: 42MB (esm) + 84MB (ViT-small) + 10MB(custom Graph model) = 152MB
4. Parameter sizes: -
   
| Component | Model | Params |
|----------|------|--------|
| **Sequence Encoder** | ProtBERT (BERT-base style) | ~420M|
| **Structural Encoder** | ViT-Small (`vit_small_patch16_224`) | ~22M |
| **Graph Transformer** | GPSConv (2 layers + SAGEConv) | ~1–3M |
| **Fusion Layer** | MultiheadAttention (320 dim, 4 heads) | ~0.4M | 
| **Classifier Head** | FFN (320 → 4) | ~1K | 
| **Total** | | 450M 

