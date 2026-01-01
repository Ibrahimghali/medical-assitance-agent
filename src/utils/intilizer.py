import hashlib
import json
from datetime import datetime
from pathlib import Path
from llama_index.core import Document
from llama_index.llms.ollama import Ollama
from utils.cleaner import normalize_text, remove_noise_from_text




def init_llm(url: str, port: int, model: str, keep_alive: int, 
            context_window: int, request_timeout: float, temperatue: float):
    
    ollama_llm = Ollama(
    model=model,
    base_url=f"http://{url}:{port}",
    keep_alive=keep_alive,          
    context_window=context_window,
    request_timeout=request_timeout,
    temperature=temperatue,
    )
    return ollama_llm



def custom_json_reader(file_path: str, version :str):
    data_path= Path(file_path)
    if not data_path.exists():
        raise FileNotFoundError(f"{file_path} not found")
    
    document= []
    data= json.loads(data_path.read_text(encoding="utf-8"))    
    for raw in data:

        # extracted_text= extract_after_introduction(raw.get('text'))
        cleaned_text= remove_noise_from_text(raw.get('text'))

        normalized_text= normalize_text(cleaned_text)
        
        doc= Document(
            text= normalized_text,
            metadata= {
                "doc_id": hashlib.sha1(raw["url"].encode()).hexdigest(),
                "title": raw.get("title"),
                "url": raw.get("url"),
                "doc_type": "medical_article",
                "source": "dermnetnz",
                "ingested_at": datetime.utcnow().isoformat(),
                "ingestion_version": version
            },
            )
        document.append(doc)
    return document


