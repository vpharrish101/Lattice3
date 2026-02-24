import os
import torch
import mlflow

from torch.utils.data import DataLoader
from data.DatasetLoader import ViT_emb_Dataset
from Engine.engine import Engine_ViT_emb
from utils import load_cfg


###
cfg=load_cfg()

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT=cfg["ViT_embed"]["data_root"]
CHECKPOINT=cfg["ViT_embed"]["checkpoint"]
###


def ViT_generate_embeddings():

    mlflow.set_experiment("ViT_embedding_gen")

    with mlflow.start_run():
        for split in ["train","val","test"]:
            pt_dir=os.path.join(DATA_ROOT,split,"Tensors")
            dataset=ViT_emb_Dataset(pt_dir)

            loader=DataLoader(
                dataset,
                batch_size=cfg["ViT_embed"]["dataloader"]["batch_size"],
                shuffle=False,
                num_workers=0)
            
            model=Engine_ViT_emb()
            checkpoint=torch.load(CHECKPOINT,map_location="cpu")
            model.load_state_dict(checkpoint)
            model.to(DEVICE)
            model.eval()

            with torch.no_grad():
                for imgs,paths in loader:

                    imgs=imgs.to(DEVICE,non_blocking=True)
                    with torch.autocast("cuda",dtype=torch.float16): embeddings=model(imgs)
                    
                    embeddings=embeddings.cpu()

                    for emb,path in zip(embeddings,paths):
                        data=torch.load(path,map_location="cpu")
                        data["vit_struct_emb"]=emb.half()
                        torch.save(data,path)



        
