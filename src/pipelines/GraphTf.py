import os
import time
import torch    
import torch.nn as nn
import mlflow
import mlflow.pytorch

from torch_geometric.loader import DataLoader
from data.DatasetLoader import GT_Dataset
from Engine.engine import Engine_TFConv
from utils import load_cfg


cfg=load_cfg()

DEVICE=torch.device("cuda")
EPOCHS=cfg["GraphTFConv"]["epochs"]
DATA_ROOT=cfg["GraphTFConv"]["data_root"]
CHECKPOINT=cfg["GraphTFConv"]["checkpoint"]



def TFConv_train():

    mlflow.set_experiment("GraphTFConv_train")

    with mlflow.start_run():

        mlflow.log_params(
            params={
                "epochs":cfg["GraphTFConv"]["epochs"],
                "lr":cfg["GraphTFConv"]["optimizer"]["lr"],
                "weight_decay":cfg["GraphTFConv"]["optimizer"]["weight_decay"],
                "train_batch_size":4,
                "val_batch_size":2,
                "model":"Engine_TFConv",
                "scheduler":"CosineAnnealingLR"
            },synchronous=True
        )

        start_time=time.time()

        trainSplit=DataLoader(
            GT_Dataset(os.path.join(DATA_ROOT,"train")),
            batch_size=4,
            shuffle=True,
            num_workers=0)

        valSplit=DataLoader(
            GT_Dataset(os.path.join(DATA_ROOT,"val")),
            batch_size=2,
            shuffle=True)

        
        model=Engine_TFConv().to(DEVICE)

        loss_fn=nn.CrossEntropyLoss()
        lr=cfg["GraphTFConv"]["lr"]

        optimizer=torch.optim.AdamW( #type:ignore
            model.parameters(),
            lr=cfg["GraphTFConv"]["optimizer"]["lr"],
            weight_decay=cfg["GraphTFConv"]["optimizer"]["weight_decay"])
        
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=EPOCHS)
        
        for epoch in range(EPOCHS):

            model.train()
            train_correct=0
            train_total=0
            train_loss=0
            
            for batch in trainSplit:
                optimizer.zero_grad()
                batch=batch.to(DEVICE)

                logits=model(
                    batch.edge_index,
                    batch.x,
                    batch.batch,
                    batch.vit_emb)            

                loss=loss_fn(logits,batch.y)
                loss.backward()
                optimizer.step()

                train_loss+=loss.item()
                preds=logits.argmax(dim=1)
                train_correct+=(preds==batch.y).sum().item()
                train_total+=batch.y.size(0)

            train_acc=train_correct/train_total

            model.eval()
            val_correct=0
            val_total=0

            with torch.no_grad():
                for batch in valSplit:

                    batch=batch.to(DEVICE)
                    logits=model(
                        batch.edge_index,
                        batch.x,
                        batch.batch,
                        batch.vit_emb)

                    preds=logits.argmax(dim=1)

                    val_correct+=(preds==batch.y).sum().item()
                    val_total+=batch.y.size(0)

            val_acc=val_correct/val_total

            mlflow.log_metric("train_loss",train_loss,step=epoch)
            mlflow.log_metric("train_acc",train_acc,step=epoch)
            mlflow.log_metric("val_acc",val_acc,step=epoch)

            scheduler.step()

        runtime=time.time()-start_time
        mlflow.log_metric("runtime_seconds",runtime)

        mlflow.pytorch.log_model(model,artifact_path="model") #type:ignore

        torch.save(model.state_dict(),CHECKPOINT)
            

'''
batch is a PyG dataset that contains: -

edge_index,
x (node features),
batch (seq)
vit_emb
'''

def TFConv_predict(batch):

    model=Engine_TFConv().to(DEVICE)
    checkpoint=torch.load(CHECKPOINT,map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()

    with torch.no_grad():
        batch=batch.to(DEVICE)
        logits=model(
            batch.edge_index,
            batch.x,
            batch.batch,
            batch.vit_emb)
        return torch.argmax(logits,dim=1)