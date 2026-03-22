import time
import mlflow

from fastapi import FastAPI,HTTPException,Request
from fastapi.concurrency import run_in_threadpool
from contextlib import asynccontextmanager 
from src.data import preprocessor
from src.pipelines import ViT_emb
from src.pipelines import ProtBERT
from src.pipelines import GraphTf

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_vit=ViT_emb.load_ViT()
    app.state.model_bert=ProtBERT.load_ProtBERT()
    app.state.model_graphtf=GraphTf.load_GraphTf()
    yield


app=FastAPI(title="Lattice3",lifespan=lifespan)
cache={}
mlflow.set_experiment("Lattice3_Inference")

def _predict_internal(model_vit,model_bert,model_graphtf,pdb_id: str):
    start=time.time()
    if pdb_id in cache:return cache[pdb_id]
    if len(pdb_id)<5: raise HTTPException(status_code=400,detail="Invalid PDB ID format")
    with mlflow.start_run():
        mlflow.log_param("pdb_id",pdb_id)

        try:
            t0=time.time()
            x=preprocessor.build_protein(
                domain_id=pdb_id,
                fasta_dir=None,
                tensor_dir=None,
                return_data=True)

            t1=time.time()

            if x is None: raise ValueError("Protein preprocessing failed")

        except Exception as e:
            mlflow.set_tag("error_stage","preprocess")
            mlflow.log_param("error_msg",str(e))
            raise HTTPException(500,f"Preprocessing failed: {str(e)}")

        try:
            x["vit_struct_emb"]=ViT_emb.ViT_embed_single(model_vit,x["V_dense_map"]) 
            t2=time.time()

        except Exception as e:
            mlflow.set_tag("error_stage","vit")
            mlflow.log_param("error_msg",str(e))
            raise HTTPException(500,f"ViT embedding failed: {str(e)}")

        try:
            x["esm_residue_emb"]=ProtBERT.ProtBERT_embed_single(model_bert,x["sequence"])
            t3=time.time()

        except Exception as e:
            mlflow.set_tag("error_stage","protbert")
            mlflow.log_param("error_msg",str(e))
            raise HTTPException(500,f"ProtBERT embedding failed: {str(e)}")

        seq_len=x["esm_residue_emb"].shape[0]
        graph_len=x["G_num_nodes"]
        min_len=min(seq_len,graph_len)
        x["esm_residue_emb"]=x["esm_residue_emb"][:min_len]
        edge_index=x["G_edge_index"]
        mask=(edge_index[0]<min_len)&(edge_index[1]<min_len)
        x["G_edge_index"]=edge_index[:,mask]
        x["G_num_nodes"]=min_len

        if len(x["vit_struct_emb"].shape)>1: x["vit_struct_emb"]=x["vit_struct_emb"][:min_len]
        if min_len<5: raise HTTPException(400,"Protein too small after trimming")

        device=next(model_graphtf.parameters()).device
        dtype=next(model_graphtf.parameters()).dtype

        x["esm_residue_emb"]=x["esm_residue_emb"].to(device=device,dtype=dtype)
        x["vit_struct_emb"]=x["vit_struct_emb"].to(device=device,dtype=dtype)
        x["G_edge_index"]=x["G_edge_index"].to(device)

        try:
            res=GraphTf.TFConv_predict(
                model=model_graphtf,
                edge_index=x["G_edge_index"],
                x=x["esm_residue_emb"],
                vit_emb=x["vit_struct_emb"]
            )
            t4=time.time()

        except Exception as e:
            mlflow.set_tag("error_stage","graph")
            mlflow.log_param("error_msg",str(e))
            raise HTTPException(500,f"GraphTF prediction failed: {str(e)}")

        timings={
            "preprocess": t1-t0,
            "vit": t2-t1,
            "protbert": t3-t2,
            "graph": t4-t3,
            "total": t4-start}

        for k,v in timings.items():
            mlflow.log_metric(k,v)

        mlflow.log_metric("prediction",int(res))
        mlflow.set_tag("status","success")

        result={
            "pdb_id": pdb_id,
            "prediction": int(res),
            "timings": {k: round(v,3) for k,v in timings.items()}}

        cache[pdb_id]=result
        return result
    


@app.get("/health")
def health_chk():
    return {"status": "OK","message": "Backend up and running"}

@app.post("/predict")
async def predict(pdb_id: str,request:Request):
    return await run_in_threadpool(_predict_internal,
                                   request.app.state.model_vit,
                                   request.app.state.model_bert,
                                   request.app.state.model_graphtf,
                                   pdb_id)