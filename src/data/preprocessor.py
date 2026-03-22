import os
import time
import torch
import pandas as pd
import mlflow
import numpy as np

from concurrent.futures import ProcessPoolExecutor
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from src.data.datautils import FASTA_seqgen,contact_mapgen
from src.utils import load_cfg

cfg=load_cfg()
DATA_ROOT=cfg["Preprocessor"]["data_root"]

DIST_THRESHOLD=cfg["Preprocessor"]["graph"]["dist_threshold"]
MAX_RESIDUES=cfg["Preprocessor"]["graph"]["max_residues"]
EDGE_CAP=cfg["Preprocessor"]["graph"]["edge_cap"]
NUM_WORKERS=4


def worker(domain_id,fasta_dir,tensor_dir):
    return build_protein(domain_id,fasta_dir,tensor_dir)

def build_protein(domain_id, 
                  fasta_dir, 
                  tensor_dir,
                  config=ProteinGraphConfig(granularity="CA"),
                  return_data=False):

    pdb_code=domain_id[:4].lower()
    chain_id=domain_id[4] 

    if not return_data: 
        if os.path.exists(os.path.join(tensor_dir,f"{domain_id}.pt")) and os.path.exists(os.path.join(fasta_dir,f"{domain_id}.fasta")):
            print(f"skipped {domain_id}")
            return "skipped"

    try:
        graph=construct_graph(
            config=config,
            pdb_code=pdb_code)
        
    except Exception:
        print(f"Failed to load {domain_id}")
        return False

    chain_nodes=[n for n in graph.nodes(data=True) if n[1].get("chain_id")==chain_id]
    sorted_nodes=sorted(chain_nodes,key=lambda x:x[1]["residue_number"])
    coords=np.array([n[1]["coords"] for n in sorted_nodes if "coords" in n[1]])
    coords=torch.from_numpy(coords).float()

    if coords.shape[0]==0: return False
    if coords.shape[0]>MAX_RESIDUES: return False

    maps=contact_mapgen(
        coords,
        pdb_id=domain_id,
        tensor_dir=tensor_dir if not return_data else None,
        return_data=return_data)
    
    if maps is False or maps is None: return None if return_data else False

    seq=FASTA_seqgen(
        sorted_nodes,
        pdb_id=domain_id,
        fasta_dir=fasta_dir if not return_data else None,
        return_data=return_data)
    
    print(f"Built {domain_id}")

    if return_data: 
        return {
            "V_dense_map": maps["V_dense_map"],#type:ignore
            "G_edge_index": maps["G_edge_index"],#type:ignore
            "G_num_nodes": maps["G_num_nodes"],#type:ignore
            "sequence": seq
        }
    return True



def preprocess():

    mlflow.set_experiment("Protein_preprocessing")

    with mlflow.start_run():

        mlflow.log_params(
            params={
                "dist_threshold":DIST_THRESHOLD,
                "max_residues":MAX_RESIDUES,
                "edge_cap":EDGE_CAP,
                "granularity":"CA"
            },synchronous=True
        )

        start_time=time.time()
        total_domains=0
        built_domains=0
        failed_domains=0

        for split in ["train","val","test"]:

            split_dir=os.path.join(DATA_ROOT,split)
            csv_path=os.path.join(split_dir,f"{split}.csv")
            fasta_dir=os.path.join(split_dir,"FASTA")
            tensor_dir=os.path.join(split_dir,"Tensors")

            os.makedirs(fasta_dir,exist_ok=True)
            os.makedirs(tensor_dir,exist_ok=True)

            df=pd.read_csv(csv_path)
            domain_ids=df["ID"].unique()

            split_total=0
            split_built=0

            with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
                results=executor.map(
                    worker,
                    domain_ids,
                    [fasta_dir]*len(domain_ids),
                    [tensor_dir]*len(domain_ids))

                for state in results:
                    total_domains+=1
                    split_total+=1
                    if state:
                        built_domains+=1
                        split_built+=1
                    else:
                        failed_domains+=1

            mlflow.log_metric(f"{split}_total_domains",split_total)
            mlflow.log_metric(f"{split}_built_domains",split_built)

        runtime=time.time()-start_time

        mlflow.log_metric("total_domains",total_domains)
        mlflow.log_metric("built_domains",built_domains)
        mlflow.log_metric("failed_domains",failed_domains)
        mlflow.log_metric("runtime_seconds",runtime)


