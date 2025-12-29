import json
import logging
import re
import time
from llama_index.core.schema import MetadataMode
from datetime import datetime
from pathlib import Path
from llama_index.core import Document
from llama_index.core import SimpleDirectoryReader, Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor, QuestionsAnsweredExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.llms.ollama import Ollama


def set_utc_log():
    logging.basicConfig(
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.Formatter.converter = time.gmtime
    return logging


def init_llm(url: str, port: int, model: str, keep_alive: int, 
            context_window: int, request_timeout: float, temparatue: float):
    
    ollama_llm = Ollama(
    model=model,
    base_url=f"http://{url}:{port}",
    keep_alive=keep_alive,          
    context_window=context_window,
    request_timeout=request_timeout,
    temperature=temparatue
    )
    return ollama_llm

def extract_after_introduction(text):

    match = re.search(r'Introduction\s*', text, flags=re.IGNORECASE)
    if match:
        return text[match.end():].strip()
    
    return text.strip()


def normalize_text(text: str)-> str:
    text= re.sub(r'\n{2,}', '\n', text)
    text= re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def custom_json_reader(file_path: str):
    data_path= Path(file_path)
    if not data_path.exists():
        raise FileNotFoundError(f"{file_path} not found")
    
    document= []
    data= json.loads(data_path.read_text(encoding="utf-8"))    
    for raw in data:

        extracted_text= extract_after_introduction(raw.get('text'))
        normalized_text= normalize_text(extracted_text)
        doc= Document(
            text= normalized_text,
            metadata= {
                "title": raw.get("title"),
                "url": raw.get("url"),
                "doc_type": "medical_article",
                "source": "dermnetnz",
                "ingested_at": datetime.utcnow().isoformat(),
                "ingestion_version": "v1.0.0"
            },
            # Exclude 'filename' from both LLM and embedding views
            excluded_llm_metadata_keys=["filename"],
            excluded_embed_metadata_keys=["filename"],
            # Customize how metadata is formatted
            metadata_separator="\n",
            metadata_template="{key}=>{value}",
            text_template="Metadata: \n{metadata_str}\n-------\n Content=>{content}"
            )
        document.append(doc)
    return document



def inspect_for_llm(doc):
    return doc.getcontent(metdata_mode=MetadataMode.LLM)

def inspect_for_embed(doc):
    return doc.getcontent(metdata_mode=MetadataMode.EMBED)



def transform(separator: str, chunk_size: int, chunk_overlap: int):
    splitter = SentenceSplitter(
        separator=separator,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    # title_extractor= TitleExtractor(
    #     llm= llm,
    #     nodes=nodenb
    # )
    # qa_extractor= QuestionsAnsweredExtractor(
    #     llm= llm,
    #     questions= question_nb
    # )

    pipeline = IngestionPipeline(
        transformations=[splitter]
    )

    return pipeline


async def run_pipeline(pipeline, docs):
    return await pipeline.arun(documents= docs, in_place= False, show_progress= True)