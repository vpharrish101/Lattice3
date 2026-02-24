import os
import time
import torch
import mlflow

from torch.utils.data import DataLoader
from transformers import AutoTokenizer,AutoModel
from data.DatasetLoader import ESM_Dataset
from Engine.engine import Engine_ProtBERT_emb
from utils import load_cfg

cfg=load_cfg()

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT=cfg["ProtBERT"]["data_root"]
MODEL_NAME=cfg["ProtBERT"]["model_name"]
BATCH_SIZE=cfg["ProtBERT"]["epochs"]


def ProtBERT_generate_embedding():

    mlflow.set_experiment("ProtBERT_embedding_gen")

    with mlflow.start_run():

        mlflow.log_params(
            params={
                "model_name":MODEL_NAME,
                "batch_size":BATCH_SIZE,
                "device":str(DEVICE),
                "num_workers":cfg["ProtBERT"]["dataloader"]["num_workers"]
            },synchronous=True
        )

        start_time=time.time()
        total_samples=0

        model=Engine_ProtBERT_emb(
            AutoTokenizer.from_pretrained(MODEL_NAME,cache_dir=r"D:\Python311\Pets\GraphT\models"),
            AutoModel.from_pretrained(MODEL_NAME,cache_dir=r"D:\Python311\Pets\GraphT\models")
        ).to(DEVICE)
        
        model.eval()

        for split in ["train","val","test"]:

            split_dir=os.path.join(DATA_ROOT,split)
            loader=DataLoader(
                ESM_Dataset(split_dir),
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=cfg["ProtBERT"]["dataloader"]["num_workers"],
                pin_memory=True,
                persistent_workers=True)

            split_count=0
            for seqs,pt_paths in loader:
                embs=model(seqs)
                for emb,pt_path in zip(embs,pt_paths):
                    if not os.path.exists(pt_path): continue
                    data=torch.load(pt_path,map_location="cpu")
                    if "esm_residue_emb" in data: continue
                    data["esm_residue_emb"]=emb
                    torch.save(data,pt_path)
                    split_count+=1
                    total_samples+=1

            mlflow.log_metric(f"{split}_samples_processed",split_count)

        runtime=time.time()-start_time
        mlflow.log_metric("total_samples_processed",total_samples)
        mlflow.log_metric("runtime_seconds",runtime)