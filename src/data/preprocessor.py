import os
import time
import torch
import pandas as pd
import mlflow

from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from datautils import FASTA_seqgen,contact_mapgen


DATA_ROOT=r"D:\Python311\Pets\GraphFold\data"

DIST_THRESHOLD=8.0
MAX_RESIDUES=600
EDGE_CAP=20000


def build_protein(domain_id, 
                  fasta_dir, 
                  tensor_dir,
                  config):

    final_path=os.path.join(tensor_dir,f"{domain_id}.pt")
    if os.path.exists(final_path):
        return False

    pdb_code=domain_id[:4].lower()
    chain_id=domain_id[4] 

    try:
        graph=construct_graph(
            config=config,
            pdb_code=pdb_code)
        
    except Exception:
        print(f"Failed to load {domain_id}")
        return False

    chain_nodes=[n for n in graph.nodes(data=True) if n[1].get("chain_id")==chain_id]
    sorted_nodes=sorted(chain_nodes,key=lambda x:x[1]["residue_number"])
    coords=torch.tensor([n[1]["coords"] for n in sorted_nodes if "coords" in n[1]],dtype=torch.float32)

    if coords.shape[0]==0: return False
    if coords.shape[0]>MAX_RESIDUES: return False

    state=contact_mapgen(coords,domain_id,tensor_dir)
    if state:
        FASTA_seqgen(sorted_nodes,domain_id,fasta_dir)
        print(f"Built {domain_id}")
        return True

    return False



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

        config=ProteinGraphConfig(granularity="CA")

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

            for domain_id in domain_ids:

                total_domains+=1
                split_total+=1

                state=build_protein(domain_id,fasta_dir,tensor_dir,config)

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