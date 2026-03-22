import os
import torch
import mlflow
import torch.nn.functional as F

from torch.utils.data import DataLoader
from src.data.DatasetLoader import ViT_emb_Dataset
from src.Engine.engine import Engine_ViT_emb
from src.utils import load_cfg


cfg=load_cfg()

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT=cfg["ViT_embed"]["data_root"]
CHECKPOINT=cfg["ViT_embed"]["checkpoint"]

def load_ViT():
    model=Engine_ViT_emb()
    checkpoint=torch.load(CHECKPOINT,map_location="cpu")
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    return model

def ViT_generate_embeddings(model,batch=None):

    mlflow.set_experiment("ViT_embedding_gen")
    with mlflow.start_run():
        
        mlflow.log_param("checkpoint_path", CHECKPOINT)
        mlflow.log_params({
            "input_size": 224,
            "channels": 3,
            "normalization": "clamp_20_div_20",
            "autocast": "fp16"})
        
        for split in ["train","val","test"]:
            pt_dir=os.path.join(DATA_ROOT,split,"Tensors")
            dataset=ViT_emb_Dataset(pt_dir)

            loader=DataLoader(
                dataset,
                batch_size=cfg["ViT_embed"]["dataloader"]["batch_size"],
                shuffle=False,
                num_workers=0)
            
            with torch.no_grad():
                for imgs,paths in loader:

                    imgs=imgs.to(DEVICE,non_blocking=True)
                    with torch.autocast("cuda",dtype=torch.float16): embeddings=model(imgs)
                    
                    embeddings=embeddings.cpu()

                    for emb,path in zip(embeddings,paths):
                        data=torch.load(path,map_location="cpu")
                        data["vit_struct_emb"]=emb.half()
                        torch.save(data,path)


def ViT_embed_single(model,dense_map: torch.Tensor):

    dense=dense_map.float().unsqueeze(0).unsqueeze(0)
    dense=F.interpolate(dense,size=(224,224),mode="bilinear",align_corners=False)
    dense=dense.repeat(1,3,1,1)
    dense=dense.clamp(max=20.0)/20.0
    dense=dense.to(DEVICE)

    with torch.no_grad():
        with torch.autocast("cuda",dtype=torch.float16):
            emb=model(dense)
            
    return emb.squeeze(0).cpu()