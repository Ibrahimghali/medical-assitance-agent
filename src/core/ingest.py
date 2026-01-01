
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
import asyncio
from llama_index.core import VectorStoreIndex, StorageContext


from utils.cleaner import run_pipeline, set_utc_log, transform
from utils.model_wrapper import ONNXEmbedding
from utils.intilizer import custom_json_reader



async def run_ingestion(data_path, version, separator, chunk_size, chunk_overlap, onnx_model_path, tokenizer_name, persistent_storage, collection_name):

    logger= set_utc_log()
    logger.info(f"Reading data from {data_path}")
    docs = custom_json_reader(data_path, version)
    pipeline = transform(separator, chunk_size, chunk_overlap)
    nodes = await run_pipeline(pipeline, docs)
    logger.info(f"Created {len(nodes)} nodes with transformations")

    embed_model = ONNXEmbedding(model_path=onnx_model_path, tokenizer_name=tokenizer_name)
    chroma_client = chromadb.PersistentClient(path=persistent_storage)
    collection  = chroma_client.get_or_create_collection(collection_name)
    
    if collection.count() > 0:
        logger.info("Index already exists. Skipping ingestion.")
        return

    vector_store = ChromaVectorStore(
        chroma_collection=collection,
        embedding_function=embed_model,
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    logger.info(f"Ingestion complete. Vector count: {collection.count()}")

