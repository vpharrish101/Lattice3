import torch
import timm

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg

from torch_geometric.nn import global_mean_pool


DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN=1024


class Engine_ViT_train(nn.Module):
    def __init__(self,ViT_model):
        super().__init__()
        self.model=ViT_model
        self.projection_head=nn.Sequential(
            nn.Linear(self.model.num_features,320),
            nn.LayerNorm(320),
            nn.GELU(),
            nn.Dropout(0.3))
        self.classification_head=nn.Linear(320,4)

    def forward(self,dense_map):
        x=self.model(dense_map)
        x=self.projection_head(x)

        return self.classification_head(x)


class Engine_ViT_emb(nn.Module):
    def __init__(self,backbone="vit_small_patch16_224",embed_dim=320):
        super().__init__()
        self.model=timm.create_model(
            backbone,
            pretrained=False, 
            num_classes=0)
        self.projection_head=nn.Sequential(
            nn.Linear(self.model.num_features,embed_dim),
            nn.LayerNorm(embed_dim))

    def forward(self,x):
        feats=self.model(x)
        return self.projection_head(feats)


class Engine_ProtBERT_emb(nn.Module):
    def __init__(self,tokenizer,model):
        super().__init__()
        self.tokenizer=tokenizer
        self.model=model

    @torch.no_grad()
    def forward(self,seqs):
        
        tokens=self.tokenizer(seqs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_LEN)
            
        tokens={k:v.to(DEVICE) for k,v in tokens.items()}

        with torch.autocast("cuda",dtype=torch.float16):
            output=self.model(**tokens)
        hidden=output.last_hidden_state
        mask=tokens["attention_mask"]
        embs=[]

        for i in range(hidden.size(0)):
            L=int(mask[i].sum())
            embs.append(hidden[i,:L].half().cpu())

        return embs


class Engine_TFConv(nn.Module):
    def __init__(self,in_channels=320):
        super().__init__()
        self.TransformerConv_1=pyg.GPSConv(
                              channels=in_channels,
                              conv=pyg.SAGEConv(in_channels,in_channels),
                              heads=4,
                              dropout=0.1,
                              act="RELU",
                              norm="layer_norm")
        self.TransformerConv_2=pyg.GPSConv(
                              channels=in_channels,
                              conv=pyg.SAGEConv(in_channels,in_channels),
                              heads=4,
                              dropout=0.1,
                              act="RELU",
                              norm="layer_norm")
        self.node_norm=nn.LayerNorm(in_channels)
        self.fusion_norm=nn.LayerNorm(in_channels)
        self.out=nn.Sequential(nn.Dropout(0.118),nn.Linear(in_channels,in_channels))
        self.CMFusion=nn.MultiheadAttention(embed_dim=320,num_heads=4,batch_first=True)
        self.classifier=nn.Linear(in_channels,4)

    def forward(self,edge_index,x,batch,vit_emb):
        
        x=self.TransformerConv_1(x,edge_index)
        x=self.TransformerConv_2(x,edge_index)
        x=self.node_norm(x)
        x=global_mean_pool(x,batch)
        
        g=self.out(x).unsqueeze(1)
        v=vit_emb.float().unsqueeze(1)

        g=F.normalize(g,dim=2)
        v=F.normalize(v,dim=2)
        attn_out,_=self.CMFusion(g,v,v)

        g=g+attn_out
        g=self.fusion_norm(g)
        g=g.squeeze(1)

        return self.classifier(g)