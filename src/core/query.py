
 
from config import BASE_URL, COLLECTION_NAME, CONTEXT_WINDOW, KEEP_ALIVE, MODEL_NAME, ONNX_MODEL_PATH, PERSISTANT_STORAGE, PORT, RERANK_MODEL_PATH, RERANK_TOKENIZER_NAME, RERANK_TOP_N, REQUEST_TIMEOUT, SYSTEM_PROMPT, TEMPERATURE, TOKENIZER_NAME, VERSION
from utils.cleaner import set_utc_log
from utils.intilizer import init_llm
from utils.model_wrapper import ONNXEmbedding, ONNXReranker
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import ( VectorStoreIndex, PromptTemplate,)
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter


def test(query_engine, query_text):
    resp = query_engine.query(query_text)
    sources = []
    for node_with_score in resp.source_nodes:
        node = node_with_score.node
        meta = node.metadata

        sources.append({
            "title": meta.get("title"),
            "url": meta.get("url"),
            "score": node_with_score.score
        })

    return {
        "answer": resp.response,
        "sources": sources
    }


def run_query(query_text, base_url, port, model_name, keep_alive, context_window, request_timeout, temperatue):
    logger = set_utc_log()
    llm = init_llm( base_url, port, model_name, keep_alive, context_window, request_timeout, temperatue)

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
    resp = test(query_engine= query_engine, query_text= query_text)
    return resp
    
    

