import os
import torch

from graphein.protein.config import ProteinGraphConfig
from Bio.Data.PDBData import protein_letters_3to1

DATA_ROOT=r"D:\Python311\Pets\GraphFold\data"

DIST_THRESHOLD=8.0
MAX_RESIDUES=600
EDGE_CAP=20000


config=ProteinGraphConfig(granularity="CA")

def FASTA_seqgen(sorted_nodes,
                 pdb_id,
                 fasta_dir,
                 return_data=False):

    sequence="".join(protein_letters_3to1.get(n[1]["residue_name"].upper(), "X") for n in sorted_nodes)
    if return_data: return sequence
    fasta_path=os.path.join(fasta_dir, f"{pdb_id}.fasta")
    with open(fasta_path, "w") as f:
        f.write(f">{pdb_id}\n")
        f.write(sequence)
    return None


def contact_mapgen(coords,
                   pdb_id, 
                   tensor_dir,
                   return_data=False):

    dist=torch.cdist(coords,coords)
    dense_map=dist.to(torch.float16)
    dense_map.fill_diagonal_(0)

    edge_index=(dist<DIST_THRESHOLD).nonzero(as_tuple=False).t()
    mask=edge_index[0]!=edge_index[1]
    edge_index=edge_index[:,mask].long()

    if edge_index.shape[1]>EDGE_CAP: return False

    data={
        "V_dense_map": dense_map,
        "G_edge_index": edge_index,
        "G_num_nodes": coords.shape[0]}
    
    if return_data: return data
    tmp_path=os.path.join(tensor_dir,f"{pdb_id}.tmp")
    final_path=os.path.join(tensor_dir,f"{pdb_id}.pt")
    torch.save(data,tmp_path)
    os.replace(tmp_path,final_path)

    return True
