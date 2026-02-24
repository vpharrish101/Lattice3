import os
import torch
import torch.nn.functional as F
import pandas as pd

from torch.utils import data as torch_data
from torch_geometric import data as pyg_data



class ViT_train_Dataset(torch_data.Dataset):

    def __init__(self,split_dir):
        tensor_dir=os.path.join(split_dir,"Tensors")
        csv_path=os.path.join(split_dir,f"{os.path.basename(split_dir)}.csv")
        df=pd.read_csv(csv_path)

        self.samples=[]
        for _,row in df.iterrows():
            pdb=row["ID"]
            label=int(row["Class"])-1
            path=os.path.join(tensor_dir,f"{pdb}.pt")
            if os.path.exists(path):
                self.samples.append((path,label))

    def __len__(self):
        return len(self.samples)    

    def __getitem__(self,idx):
        path,label=self.samples[idx]
        data=torch.load(path,map_location="cpu")
        dense=data["V_dense_map"].float()   

        dense=dense.unsqueeze(0).unsqueeze(0)
        dense=F.interpolate(
            dense,
            size=(224,224),
            mode="bilinear",
            align_corners=False)

        dense=dense.squeeze(0)     
        dense=dense.repeat(3,1,1)  
        dense=torch.exp(-dense/10.0)

        return dense,torch.tensor(label).long()
    


class ViT_emb_Dataset(torch_data.Dataset):

    def __init__(self,pt_dir):
        self.files=[
            os.path.join(pt_dir,f)
            for f in os.listdir(pt_dir)
            if f.endswith(".pt")
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):

        path=self.files[idx]
        data=torch.load(path,map_location="cpu")

        dense=data["V_dense_map"].float()
        dense=dense.unsqueeze(0).unsqueeze(0)
        dense=F.interpolate(
            dense,
            size=(224,224),
            mode="bilinear",
            align_corners=False)

        dense=dense.squeeze(0)
        dense=dense.repeat(3,1,1)

        dense=dense.clamp(max=20.0)/20.0

        return dense,path



class ESM_Dataset(torch_data.Dataset):
    
    def __init__(self,
                 split_dir):
        self.fasta_dir=os.path.join(split_dir,"FASTA")
        self.tensor_dir=os.path.join(split_dir,"Tensors")
        self.files=[
            f for f in os.listdir(self.fasta_dir)
            if f.endswith(".fasta")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):

        fasta_name=self.files[idx]
        pdb=fasta_name.replace(".fasta","")

        fasta_path=os.path.join(self.fasta_dir,fasta_name)
        pt_path=os.path.join(self.tensor_dir,f"{pdb}.pt")

        with open(fasta_path) as f:
            seq="".join(
                line.strip()
                for line in f
                if not line.startswith(">")
            )
        return seq,pt_path



class GT_Dataset(pyg_data.Dataset):

    def __init__(self,split_dir):
        super().__init__()
        tensor_dir=os.path.join(split_dir,"Tensors")
        csv_file=os.path.join(split_dir,f"{os.path.basename(split_dir)}.csv")
        print(csv_file)
        df=pd.read_csv(csv_file)

        self.samples=[]
        for _,row in df.iterrows():

            pdb=row["ID"]
            label=int(row["Class"])-1
            path=os.path.join(tensor_dir,f"{pdb}.pt")
            if os.path.exists(path):
                self.samples.append((path,label))

    def len(self):
        return len(self.samples)
    
    def get(self,idx):
        path, label=self.samples[idx]
        data=torch.load(path, map_location="cpu")

        edge_index=data["G_edge_index"].long()
        node_attr=data["esm_residue_emb"].float()
        vit_emb=data["vit_struct_emb"]

        data=pyg_data.Data(
            edge_index=edge_index,
            x=node_attr,
            y=torch.tensor(label,dtype=torch.long))
        
        data.vit_emb=vit_emb.view(1,-1)
        return data