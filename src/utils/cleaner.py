import logging
import re
import time
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline



def normalize_text(text: str)-> str:
    text= re.sub(r'\n{2,}', '\n', text)
    text= re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def remove_noise_from_text(text: str) -> str:
    # Patterns to remove
    patterns_to_remove = [
        # Books references
        r'Books about skin diseases',
        r'Books about the skin',
        r'Dermatology Made Easy\s*[-–—]\s*second edition',
        
        # Other recommended articles
        r'Other recommended articles',
        
        # See more images
        r'See more images',
        
        # Navigation headers (common at the beginning)
        r'Common skin conditions',
        r'Join DermNet PRO',
        r'Try our skin symptom checker',
        
        # YouTube and video references
        r'YouTube',
        r'\[YouTube\]',
        r'—\s*DermNet\s+e-lecture\s*\[YouTube\]',
        r'DermNet\s+e-lecture',
        
        # External link sections (often at the end)
        r'Useful dermatology resources for health professionals',
        r'Worldwide dermatology links for doctors',
        r'Medical journals for the dermatologist',
        r'Worldwide dermatology societies',
        r'Dermatology conferences',
        r'Worldwide dermatology links for patients',
        r'Dermatology basics:.*?\[YouTube\]',
        
        # Organization references at the end
        r'New Zealand Dermatological Society Inc\.',
        r'The Australasian College of Dermatologists',
        r'American Academy of Dermatology Association',
        r'The British Association of Dermatologists',
        
        # Common footer patterns
        r'Available here',
        r'Accessed \d{4}',
    ]
    
    cleaned_text = text
    
    # Remove each pattern
    for pattern in patterns_to_remove:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
    
    # Remove repeated navigation-like headers (e.g., "Author: ... Reviewer: ...")
    # This pattern catches lines that are typically metadata/navigation
    cleaned_text = re.sub(r'Author:\s*[^\n]+', '', cleaned_text)
    cleaned_text = re.sub(r'Reviewer:\s*[^\n]+', '', cleaned_text)
    cleaned_text = re.sub(r'Editor:\s*[^\n]+', '', cleaned_text)
    cleaned_text = re.sub(r'Reviewing dermatologist:\s*[^\n]+', '', cleaned_text)
    cleaned_text = re.sub(r'Copy edited by\s*[^\n]+', '', cleaned_text)
    cleaned_text = re.sub(r'Previous contributors:\s*[^\n]+', '', cleaned_text)
    cleaned_text = re.sub(r'Updated by\s*[^\n]+', '', cleaned_text)
    cleaned_text = re.sub(r'Edited by\s*[^\n]+', '', cleaned_text)
    
    # Clean up multiple newlines and spaces that might result from removals
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)
    
    return cleaned_text.strip()




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



def set_utc_log():
    logging.basicConfig(
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.Formatter.converter = time.gmtime
    return logging