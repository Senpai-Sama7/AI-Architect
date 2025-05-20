#!/usr/bin/env python3
"""
Embedding Generator Module

This module provides utilities for generating vector embeddings from text using
different embedding models. It supports OpenAI, local models, and provides
a unified interface for embedding generation.
"""

import asyncio
import logging
import time
from typing import List, Optional, Dict, Any, Union

import numpy as np
from openai import AsyncOpenAI
from prometheus_client import Counter, Histogram

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Prometheus metrics
EMBEDDING_OPERATIONS = Counter(
    "embedding_operations_total",
    "Total number of embedding operations",
    ["model", "status"]
)
EMBEDDING_LATENCY = Histogram(
    "embedding_latency_seconds",
    "Embedding operation latency in seconds",
    ["model"],
    buckets=[0.1, 0.5, 1, 2, 5, 10]
)

class EmbeddingGenerator:
    """
    Utility for generating vector embeddings from text.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        batch_size: int = 16,
        dimensions: Optional[int] = None,
    ):
        """
        Initialize the embedding generator.

        Args:
            model: The embedding model to use
            api_key: OpenAI API key
            batch_size: Maximum batch size for embedding generation
            dimensions: Optional output dimensionality (if supported by model)
        """
        self.model = model
        self.api_key = api_key
        self.batch_size = batch_size
        self.dimensions = dimensions
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=api_key)
        
        logger.info(f"Initialized embedding generator with model: {model}")

    async def get_embeddings(
        self,
        texts: List[str],
        normalize: bool = True,
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed
            normalize: Whether to normalize the embeddings to unit length

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
            
        start_time = time.time()
        
        try:
            # Process in batches to avoid rate limits
            all_embeddings = []
            
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                batch_result = await self._embed_batch(batch)
                all_embeddings.extend(batch_result)
                
                if i + self.batch_size < len(texts):
                    # Add a small delay between batches to avoid rate limits
                    await asyncio.sleep(0.5)
            
            if normalize:
                all_embeddings = [self._normalize_vector(emb) for emb in all_embeddings]
            
            logger.info(f"Generated {len(all_embeddings)} embeddings with model {self.model}")
            EMBEDDING_OPERATIONS.labels(model=self.model, status="success").inc()
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            EMBEDDING_OPERATIONS.labels(model=self.model, status="error").inc()
            raise
        finally:
            EMBEDDING_LATENCY.labels(model=self.model).observe(time.time() - start_time)

    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts using the configured model.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        kwargs = {"dimensions": self.dimensions} if self.dimensions else {}
        
        # Use OpenAI embeddings API
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts,
            **kwargs
        )
        
        # Extract and return the embeddings
        embeddings = [item.embedding for item in response.data]
        return embeddings

    def _normalize_vector(self, vector: List[float]) -> List[float]:
        """
        Normalize a vector to unit length.

        Args:
            vector: Input vector

        Returns:
            Normalized vector
        """
        array = np.array(vector)
        norm = np.linalg.norm(array)
        if norm > 0:
            normalized = array / norm
            return normalized.tolist()
        return vector

    async def similarity_score(
        self,
        text1: str,
        text2: str,
        method: str = "cosine",
    ) -> float:
        """
        Calculate similarity between two texts.

        Args:
            text1: First text
            text2: Second text
            method: Similarity method ('cosine', 'dot', or 'euclidean')

        Returns:
            Similarity score between 0 and 1
        """
        embeddings = await self.get_embeddings([text1, text2])
        
        if len(embeddings) != 2:
            raise ValueError("Failed to generate embeddings for comparison")
        
        vec1 = np.array(embeddings[0])
        vec2 = np.array(embeddings[1])
        
        if method == "cosine":
            # Cosine similarity: dot(v1, v2) / (||v1|| * ||v2||)
            # Since vectors are normalized, this is just the dot product
            return float(np.dot(vec1, vec2))
        elif method == "dot":
            # Dot product
            return float(np.dot(vec1, vec2))
        elif method == "euclidean":
            # Euclidean distance converted to similarity (0-1)
            distance = np.linalg.norm(vec1 - vec2)
            # Convert distance to similarity (closer to 1 means more similar)
            return float(1.0 / (1.0 + distance))
        else:
            raise ValueError(f"Unknown similarity method: {method}")


async def demo():
    """Run a simple demonstration of the embedding generator."""
    # Initialize the embedding generator
    embedding_gen = EmbeddingGenerator(
        model="text-embedding-3-small",
        dimensions=1536,  # Using explicit dimensions for the model
    )
    
    # Sample texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "The five boxing wizards jump quickly.",
        "How vexingly quick daft zebras jump!",
        "Sphinx of black quartz, judge my vow.",
        "Pack my box with five dozen liquor jugs.",
    ]
    
    # Generate embeddings
    embeddings = await embedding_gen.get_embeddings(texts)
    
    # Print embedding dimensions
    print(f"Generated {len(embeddings)} embeddings, each with {len(embeddings[0])} dimensions")
    
    # Calculate similarity between first and all other texts
    print("\nSimilarity to first text:")
    for i, text in enumerate(texts[1:], 1):
        similarity = await embedding_gen.similarity_score(texts[0], text)
        print(f"{i}. '{text[:30]}...' - Similarity: {similarity:.4f}")
    
    # Compare different similarity methods
    text_a = "I love machine learning and artificial intelligence."
    text_b = "AI and ML are fascinating technologies."
    
    print("\nComparing similarity methods:")
    for method in ["cosine", "dot", "euclidean"]:
        similarity = await embedding_gen.similarity_score(text_a, text_b, method=method)
        print(f"{method}: {similarity:.4f}")


if __name__ == "__main__":
    asyncio.run(demo())
