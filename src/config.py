


DATA_PATH= "data/raw/articles.json"
MODEL_NAME= "llama3.2:latest"
BASE_URL= "localhost"
PORT= 11434
KEEP_ALIVE= 0
CONTEXT_WINDOW= 1024
REQUEST_TIMEOUT= 300.0
TEMPERATURE= 0.1

SEPARATOR= " "
CHUNK_SIZE= 512
CHUNK_OVERLAP= 80
NODES= 2
QUESTIONS= 3
VERSION= "v0.0.2"

SIMILARITY_TOP_K= 3
RERANK_TOP_N = 3

SYSTEM_PROMPT=  """"
You are a medical question answering assistant.

You must answer strictly and only using the provided context.
Do not use prior knowledge or assumptions.
If the context does not contain the answer, respond with:
'I don’t have enough information in the provided documents to answer this question.'

Do not guess, speculate, or fill in missing details. 
"""


COLLECTION_NAME= "medical_articles"
PERSISTANT_STORAGE= "./chroma_db"


ONNX_MODEL_PATH = r"C:/Users/Ibrahim/Documents/WORK/Faculty-Projects/nlp/model/BAAI/model.onnx"
TOKENIZER_NAME = "BAAI/bge-small-en-v1.5"

RERANK_MODEL_PATH = r"C:/Users/Ibrahim/Documents/WORK/Faculty-Projects/nlp/model/reranking/model.onnx"
RERANK_TOKENIZER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"