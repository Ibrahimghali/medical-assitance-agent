

import numpy as np
import onnxruntime
from transformers import AutoTokenizer
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from pydantic import PrivateAttr
from typing import List, Optional


def get_text_embedding(texts, session, tokenizer):
    """
    Get embeddings from an ONNX model.
    
    Args:
        texts: Single string or list of strings to embed
        session: ONNX runtime inference session
        tokenizer: HuggingFace tokenizer instance
    
    Returns:
        numpy array of embeddings
    """
    if isinstance(texts, str):
        texts = [texts]

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="np")

    outputs = session.run(
        None,
        {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "token_type_ids": inputs.get("token_type_ids", np.zeros_like(inputs["input_ids"]))
        }
    )

    # Mean pooling
    token_embeddings = outputs[0]
    attention_mask = inputs["attention_mask"]
    input_mask_expanded = np.expand_dims(attention_mask, -1).astype(float)
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
    
    return sum_embeddings / sum_mask


class ONNXEmbedding(BaseEmbedding):
    """Generic ONNX embedding wrapper for LlamaIndex."""
    
    _session: onnxruntime.InferenceSession = PrivateAttr()
    _tokenizer: AutoTokenizer = PrivateAttr()
    
    def __init__(
        self,
        model_path: str,
        tokenizer_name: str,
        **kwargs
    ):
        """
        Initialize ONNX embedding model.
        
        Args:
            model_path: Path to the ONNX model file
            tokenizer_name: HuggingFace tokenizer name or path
            **kwargs: Additional arguments passed to BaseEmbedding
        """
        super().__init__(**kwargs)
        
        # Load model and tokenizer using private attributes
        self._session = onnxruntime.InferenceSession(model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def _get_query_embedding(self, query: str):
        return get_text_embedding(query, self._session, self._tokenizer)[0].tolist()
    
    def _get_text_embedding(self, text: str):
        return get_text_embedding(text, self._session, self._tokenizer)[0].tolist()
    
    async def _aget_query_embedding(self, query: str):
        return self._get_query_embedding(query)
    
    def _get_text_embeddings(self, texts: list[str]):
        return get_text_embedding(texts, self._session, self._tokenizer).tolist()


class ONNXReranker(BaseNodePostprocessor):
    """ONNX-based reranker for LlamaIndex."""
    
    _session: onnxruntime.InferenceSession = PrivateAttr()
    _tokenizer: AutoTokenizer = PrivateAttr()
    top_n: int = 3
    
    def __init__(
        self,
        model_path: str,
        tokenizer_name: str,
        top_n: int = 3,
        **kwargs
    ):
        """
        Initialize ONNX reranker.
        
        Args:
            model_path: Path to the ONNX reranking model file
            tokenizer_name: HuggingFace tokenizer name or path
            top_n: Number of top results to return after reranking
            **kwargs: Additional arguments passed to BaseNodePostprocessor
        """
        super().__init__(top_n=top_n, **kwargs)
        
        # Load model and tokenizer
        self._session = onnxruntime.InferenceSession(model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """
        Rerank nodes based on relevance to the query.
        
        Args:
            nodes: List of nodes with scores
            query_bundle: Query information
            
        Returns:
            Reranked and filtered list of nodes
        """
        if query_bundle is None or len(nodes) == 0:
            return nodes
        
        query_text = query_bundle.query_str
        
        # Create query-document pairs
        pairs = [[query_text, node.node.get_content()] for node in nodes]
        
        # Tokenize pairs
        inputs = self._tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="np",
            max_length=512
        )
        
        # Get relevance scores from ONNX model
        outputs = self._session.run(
            None,
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "token_type_ids": inputs.get("token_type_ids", np.zeros_like(inputs["input_ids"]))
            }
        )
        
        # Extract logits (usually first output)
        logits = outputs[0]
        
        # For cross-encoders, typically take the positive class score
        # or just use the raw logits if it's a regression model
        if logits.shape[-1] == 1:
            scores = logits.squeeze(-1)
        else:
            scores = logits[:, 0]  # or [:, 1] depending on model
        
        # Update node scores
        for node, score in zip(nodes, scores):
            node.score = float(score)
        
        # Sort by score and return top_n
        nodes.sort(key=lambda x: x.score or 0.0, reverse=True)
        return nodes[:self.top_n]
