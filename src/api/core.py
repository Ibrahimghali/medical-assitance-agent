from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from config import *


from core.ingest import run_ingestion
from core.query import run_query
from utils.cleaner import set_utc_log  # import the refactored function

logger = set_utc_log()

router = APIRouter()

class IngestRequest(BaseModel):
    data_path: str = DATA_PATH
    version: str = VERSION
    separator: str = SEPARATOR
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    onnx_model_path: str = ONNX_MODEL_PATH
    tokenizer_name: str = TOKENIZER_NAME
    persistent_storage: str = PERSISTANT_STORAGE
    collection_name: str = COLLECTION_NAME

class QueryRequest(BaseModel):
    query: str
    base_url: str=  BASE_URL
    port: int= PORT
    model_name: str= MODEL_NAME
    keep_alive: int= KEEP_ALIVE
    context_window: int= CONTEXT_WINDOW
    request_timeout: float= REQUEST_TIMEOUT
    temperatue: float= TEMPERATURE

# ----------------------
# API endpoint
# ----------------------
@router.post("/ingest")
async def ingest(request: IngestRequest):
    """
    Trigger ingestion of documents into the RAG system.
    """
    try:
        await run_ingestion(
            data_path=request.data_path,
            version=request.version,
            separator=request.separator,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            onnx_model_path=request.onnx_model_path,
            tokenizer_name=request.tokenizer_name,
            persistent_storage=request.persistent_storage,
            collection_name=request.collection_name
        )
        return {"status": "success", "message": "Ingestion completed."}
    except Exception as e:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=str(e))
    

@router.post("/query")
def query(request: QueryRequest):

    try:
        response = run_query(
            query_text=request.query,
            base_url= request.base_url,
            port= request.port,
            model_name= request.model_name,
            keep_alive= request.keep_alive,
            context_window= request.context_window,
            request_timeout= request.request_timeout,
            temperatue= request.temperatue
        )
        return {
            "status": "success",
            "answer": response.get("answer"),
            "sources": response.get("sources", [])
        }
    except Exception as e:
        logger.exception("Query Failed")
        raise HTTPException(status_code=500, detail=str(e))
