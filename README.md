# Doctor Assistant RAG System

A Retrieval-Augmented Generation (RAG) system powered by LlamaIndex, designed to assist doctors with medical information retrieval and analysis. This project is part of the Data Science subject project.

##  Overview

This system provides an RAG-powered medical assistant that leverages RAG technology to retrieve relevant medical information from a knowledge base and generate contextual responses using Large Language Models (LLMs).

##  Features

- **RAG-based Information Retrieval**: Combines vector embeddings with LLM reasoning
- **Semantic Search**: Powered by HuggingFace embeddings with ONNX optimization
- **Reranking**: Enhanced result relevance using reranking models
- **Vector Database**: ChromaDB for efficient similarity search
- **LLM Integration**: Supports Ollama and Groq LLMs
- **Dockerized**: Containerized deployment with Docker Compose
- **GPU Support**: NVIDIA GPU acceleration for Ollama


##  Tech Stack

- **Framework**: LlamaIndex
- **Vector Store**: ChromaDB
- **Embeddings**: HuggingFace (ONNX optimized)
- **LLMs**: Ollama, Groq
- **API**: FastAPI + Uvicorn
- **Language**: Python 3.13
- **Deployment**: Docker + Docker Compose

## Prerequisites

- Python 3.13+
- Docker and Docker Compose
- NVIDIA GPU with NVIDIA Container Toolkit (optional, for GPU acceleration)
- External Docker volumes: `llm-data`
- External Docker network: `rag-net`

### Setup Docker Resources

```bash
# Create external volume for LLM data
docker volume create llm-data

# Create external network
docker network create rag-net

mkdir data
under data you can upload the document reference
```

##  Installation

### Usage : you can use it locally also if you don't wanna use docker

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ibrahimghali/medical-assitance-agent.git
   cd medical-assitance-agent/docker
   ```

2. **Build and start services**
   ```bash
   cd docker
   docker-compose up -d
   ```

2. **Access the services**
   - RAG API: http://localhost:8000
   - Ollama: http://localhost:11434
   - API Documentation: http://localhost:8000/docs

##  API Endpoints

### Base URL
```
http://localhost:8000/api
```

### Available Endpoints
- **POST /ingest**: Ingest documents into the vector store
- **POST /query**: Query the RAG system
- **GET /health**: Health check endpoint

### Example Usage

```bash
# Query the RAG system
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the symptoms of diabetes?"}'
```

##  Notebooks

Jupyter notebooks for exploration and analysis:
- [explore.ipynb](notebook/explore.ipynb): Data exploration
- [clean_data.ipynb](notebook/clean_data.ipynb): Data cleaning pipeline

##  Development

### Project Structure
- **api/**: REST API endpoints and request/response models
- **core/**: RAG implementation (document ingestion and querying)
- **utils/**: Helper functions and model wrappers
