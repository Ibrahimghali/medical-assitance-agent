
 
from config import BASE_URL, COLLECTION_NAME, CONTEXT_WINDOW, KEEP_ALIVE, MODEL_NAME, ONNX_MODEL_PATH, PERSISTANT_STORAGE, PORT, RERANK_MODEL_PATH, RERANK_TOKENIZER_NAME, RERANK_TOP_N, REQUEST_TIMEOUT, SYSTEM_PROMPT, TEMPERATURE, TOKENIZER_NAME, VERSION
from model_wrapper import ONNXEmbedding, ONNXReranker
from utils import init_llm, set_utc_log
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import ( VectorStoreIndex, PromptTemplate,)
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

def test(query_engine):
    test_queries = [
        "What causes acne?",
        "How to treat athlete's foot?"
    ]
    for q in test_queries:
        resp = query_engine.query(q)
        print("\n==============================")
        print("QUESTION:", q)
        print("ANSWER:\n", resp.response)
        print("SOURCES:")
        for node in getattr(resp, "source_nodes", []):
            print("-", node.node.metadata.get("title"), node.node.metadata.get("url"))

def main():
    logger = set_utc_log()
    llm = init_llm( BASE_URL, PORT, MODEL_NAME, KEEP_ALIVE, CONTEXT_WINDOW, REQUEST_TIMEOUT, TEMPERATURE)

    embed_model = ONNXEmbedding(model_path=ONNX_MODEL_PATH, tokenizer_name=TOKENIZER_NAME)

    chroma_client = chromadb.PersistentClient(path=PERSISTANT_STORAGE)
    collection  = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(
        chroma_collection=collection,
        embedding_function=embed_model
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model
    )

    logger.info(f"Vector count: {collection.count()}")


    filters= MetadataFilters(filters= [
        ExactMatchFilter(key= 'source', value="dermnetnz"),
        ExactMatchFilter(key="ingestion_version", value=VERSION)
    ])

    retriever = index.as_retriever(
        similarity_top_k=5,
        filters=filters
    )

    reranker = ONNXReranker(
        model_path=RERANK_MODEL_PATH,
        tokenizer_name=RERANK_TOKENIZER_NAME,
        top_n=RERANK_TOP_N
    )

    qa_promot= PromptTemplate(
        SYSTEM_PROMPT+ """
        Context:
        {context_str}

        Question:
        {query_str}

        Answer:
        """

    )

    response_synthesizer= get_response_synthesizer(
        llm= llm,
        response_mode= ResponseMode.COMPACT,
        text_qa_template= qa_promot,
    )

    query_engine= RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[reranker],
        response_synthesizer= response_synthesizer
    )

    logger.info("Query engine ready for use")
    
    test(query_engine)


if __name__ == "__main__":
    main()