from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import asyncio
import nest_asyncio
from llama_index.core import VectorStoreIndex, StorageContext

from config import (
    BASE_URL, CHUNK_OVERLAP, CHUNK_SIZE, CONTEXT_WINDOW, DATA_PATH, 
    KEEP_ALIVE, MODEL_NAME, PORT, REQUEST_TIMEOUT, SEPARATOR, TEMPERATURE
)
from utils import custom_json_reader, init_llm, run_pipeline, set_utc_log, transform
nest_asyncio.apply()

def main():
    logger = set_utc_log()
    llm = init_llm(
        BASE_URL, PORT, MODEL_NAME, 
        KEEP_ALIVE, CONTEXT_WINDOW, REQUEST_TIMEOUT, TEMPERATURE
    )

    logger.info(f"Reading data from {DATA_PATH}")
    docs = custom_json_reader(DATA_PATH)


    pipeline = transform(SEPARATOR, CHUNK_SIZE, CHUNK_OVERLAP)
    nodes = asyncio.get_event_loop().run_until_complete(run_pipeline(pipeline, docs))
    logger.info(f"Created {len(nodes)} nodes with transformations")

    hf_embeddings = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("quickstart")

    # Create vector store with embedding function
    vector_store = ChromaVectorStore(
        chroma_collection=chroma_collection,
        embedding_function=hf_embeddings
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build persistent index only if collection is empty
    if chroma_collection.count() == 0:
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=hf_embeddings
        )
        logger.info(f"Index built with {len(nodes)} nodes")
    else:
        index = VectorStoreIndex(
            storage_context=storage_context,
            embed_model=hf_embeddings
        )
        logger.info(f"Loaded existing index with {chroma_collection.count()} vectors")

    # Create query engine
    query_engine = index.as_query_engine(llm=llm, response_mode="tree_summarize", include_text=True)
    
    logger.info("Query engine ready for use")
    # Example test queries
    test_queries = [
        "What causes acne?",
        "How to treat athlete's foot?",
        "Symptoms of cellulitis",
        "Cold sore prevention methods"
    ]

    # for q in test_queries:
    #     resp = query_engine.query(q)
    #     print(resp.source_nodes)

    for q in test_queries:
        resp = query_engine.query(q)
        print("Answer:\n", resp.response)  # the LLM answer
        print("\nSources:")
        for node in getattr(resp, "source_nodes", []):
            print(node.node.metadata.get("title"), node.node.metadata.get("url"))


if __name__ == "__main__":
    main()
