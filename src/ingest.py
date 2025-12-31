
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
import asyncio
from llama_index.core import VectorStoreIndex, StorageContext

from config import CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION_NAME, DATA_PATH, EMBEDDING_MODEL, EMBEDDING_MODEL_TOKENIZER_NAME, PERSISTANT_STORAGE, SEPARATOR, VERSION
from model_wrapper import ONNXEmbedding
from utils import custom_json_reader, run_pipeline, set_utc_log, transform


def main():
    logger= set_utc_log()
    logger.info(f"Reading data from {DATA_PATH}")
    docs = custom_json_reader(DATA_PATH, VERSION)
    pipeline = transform(SEPARATOR, CHUNK_SIZE, CHUNK_OVERLAP)
    nodes = asyncio.get_event_loop().run_until_complete(run_pipeline(pipeline, docs))
    logger.info(f"Created {len(nodes)} nodes with transformations")

    embed_model = ONNXEmbedding(model_path=EMBEDDING_MODEL, tokenizer_name=EMBEDDING_MODEL_TOKENIZER_NAME)
    chroma_client = chromadb.PersistentClient(path=PERSISTANT_STORAGE)
    collection  = chroma_client.get_or_create_collection(COLLECTION_NAME)
    
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


if __name__ == "__main__":
    main()
