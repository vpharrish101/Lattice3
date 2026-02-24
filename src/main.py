from data import preprocessor
from pipelines import ViT_train
from pipelines import ViT_emb
from pipelines import ProtBERT
from pipelines import GraphTf

"""
How this architecture works: -
    1. The preprocessing script genrates the required data (check README)

    2. ViT is finetuned with the dense contact maps, preferraly by unfreezing last 2 or 3 blocks.
    3. ViT is now used to generate dense embeddings from contact maps(dense)

    4. ProtBERT is used to generate dense embeddings from contact maps(dense) per residue (NOT POOLED)

    5. Graph Transformer block (TransformerConv) takes in ProtBERT embeddings as edge index, sparse contact map 
    as node_features, and the labels, and produces a 320 dim embedding

    6. ViT and TFConv's embeddings are cross queried (Graph=Q,ViT=K/V) and the final output is logits from FFN.
"""
def main():
    preprocessor.preprocess()
    ViT_train.ViT_finetune()
    ViT_emb.ViT_generate_embeddings()
    ProtBERT.ProtBERT_generate_embedding()
    GraphTf.TFConv_train()
