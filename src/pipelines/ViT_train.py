import os
import time
import torch
import timm
import yaml
import mlflow
import mlflow.pytorch
import torch.nn as nn

from torch.utils.data import DataLoader
from data.DatasetLoader import ViT_train_Dataset
from Engine.engine import Engine_ViT_train
from utils import load_cfg


###
cfg=load_cfg()

DATA_ROOT=cfg["ViT_train"]["data_root"]
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS=cfg["ViT_train"]["epochs"]
CHECKPOINT=cfg["ViT_train"]["checkpoint"]
###


def ViT_finetune():

    mlflow.set_experiment("ViT_finetune")

    with mlflow.start_run():

        mlflow.log_params(
            params={
                "epochs":cfg["ViT_train"]["epochs"],
                "lr":cfg["ViT_train"]["optimizer"]["lr"],
                "weight_decay":cfg["ViT_train"]["optimizer"]["weight_decay"],
                "backbone":"vit_small_patch16_224",
                "unfreeze_blocks":cfg["ViT_train"]["backbone"]["unfreeze_last_blocks"]
            },synchronous=True
        )

        start_time=time.time()

        ViT_model=timm.create_model("vit_small_patch16_224",pretrained=True,num_classes=0)

        for p in ViT_model.parameters():
            p.requires_grad=False

        n=cfg["ViT_train"]["backbone"]["unfreeze_last_blocks"]
        for block in ViT_model.blocks[-n:]:
            for p in block.parameters():
                p.requires_grad=True

        model=Engine_ViT_train(ViT_model).to(DEVICE)
        loss_fn=nn.CrossEntropyLoss()

        optimizer=torch.optim.AdamW( #type:ignore
            filter(lambda p:p.requires_grad,model.parameters()),
            lr=cfg["ViT_train"]["optimizer"]["lr"],
            weight_decay=cfg["ViT_train"]["optimizer"]["weight_decay"])

        train_loader=DataLoader(
            ViT_train_Dataset(os.path.join(DATA_ROOT,"train")),
            batch_size=cfg["ViT_train"]["dataloader"]["train_batch_size"],
            shuffle=True)

        val_loader=DataLoader(
            ViT_train_Dataset(os.path.join(DATA_ROOT,"val")),
            batch_size=cfg["ViT_train"]["dataloader"]["val_batch_size"],
            shuffle=False)

        for epoch in range(EPOCHS):

            model.train()

            total_loss=0
            correct=0
            total=0

            for dense_map,labels in train_loader:

                dense_map=dense_map.to(DEVICE,non_blocking=True)
                labels=labels.to(DEVICE)

                optimizer.zero_grad()

                logits=model(dense_map)
                loss=loss_fn(logits,labels)

                loss.backward()
                optimizer.step()

                total_loss+=loss.item()
                preds=logits.argmax(dim=1)
                correct+=(preds==labels).sum().item()
                total+=labels.size(0)

            train_acc=correct/total

            model.eval()

            val_correct=0
            val_total=0

            with torch.no_grad():

                for dense_map,labels in val_loader:

                    dense_map=dense_map.to(DEVICE,non_blocking=True)
                    labels=labels.to(DEVICE)

                    logits=model(dense_map)
                    preds=logits.argmax(dim=1)

                    val_correct+=(preds==labels).sum().item()
                    val_total+=labels.size(0)

            val_acc=val_correct/val_total

            mlflow.log_metric("train_loss",total_loss,step=epoch)
            mlflow.log_metric("train_acc",train_acc,step=epoch)
            mlflow.log_metric("val_acc",val_acc,step=epoch)

        runtime=time.time()-start_time
        mlflow.log_metric("runtime_seconds",runtime)

        mlflow.pytorch.log_model(model,artifact_path="model") #type:ignore

        w=model.state_dict()
        w.pop("classification_head.weight")
        w.pop("classification_head.bias")
        torch.save(w,CHECKPOINT)


