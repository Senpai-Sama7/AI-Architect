#!/usr/bin/env python3
"""
Vector Database Client for Knowledge Management

This module provides a wrapper around Qdrant vector database for storing and 
retrieving vector embeddings. It supports knowledge base management with 
vector similarity search capabilities.

Key features:
- Collection management (create, delete, list)
- Vector storage with metadata
- Similarity search with filtering
- Batched upsert operations
- Scrolling through collections
- Metrics collection for performance monitoring
"""

import asyncio
import json
import logging
import os
import time
from typing import Dict, List, Optional, Any, Union, Tuple

import numpy as np
from prometheus_client import Counter, Histogram
from qdrant_client import QdrantClient, models
from qdrant_client.http import models as rest
from qdrant_client.http.exceptions import UnexpectedResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Prometheus metrics
VECTOR_STORE_OPERATIONS = Counter(
    "vector_store_operations_total",
    "Total number of vector store operations",
    ["operation_type"]
)
VECTOR_STORE_LATENCY = Histogram(
    "vector_store_latency_seconds",
    "Vector store operation latency in seconds",
    ["operation_type"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1, 5]
)


class VectorStore:
    """
    Vector database client for embedding storage and retrieval.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        https: bool = False,
        grpc_port: Optional[int] = None,
        prefer_grpc: bool = True,
    ):
        """
        Initialize the vector store client.

        Args:
            host: Qdrant server host
            port: Qdrant server HTTP port
            api_key: API key for authentication
            https: Whether to use HTTPS
            grpc_port: gRPC port if different from HTTP port+1
            prefer_grpc: Whether to prefer gRPC over HTTP
        """
        self.client = QdrantClient(
            host=host,
            port=port,
            api_key=api_key,
            https=https,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
        )
        logger.info(f"Connected to Qdrant at {host}:{port}")

    async def create_collection(
        self,
        collection_name: str,
        vector_size: int = 1536,  # Default size for OpenAI embeddings
        distance: str = "Cosine",
        on_disk_payload: bool = True,
    ) -> bool:
        """
        Create a new collection for storing embeddings.

        Args:
            collection_name: Name of the collection
            vector_size: Dimensionality of vectors
            distance: Distance function ('Cosine', 'Euclid', or 'Dot')
            on_disk_payload: Store payload on disk instead of in memory

        Returns:
            True if collection was created successfully
        """
        start_time = time.time()
        try:
            # Check if collection already exists
            collections = self.client.get_collections().collections
            if any(c.name == collection_name for c in collections):
                logger.info(f"Collection {collection_name} already exists")
                return True

            # Create vector config
            vector_config = models.VectorParams(
                size=vector_size,
                distance=models.Distance(distance),
                on_disk=on_disk_payload,
            )

            # Create collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=vector_config,
            )
            
            logger.info(f"Created collection {collection_name}")
            VECTOR_STORE_OPERATIONS.labels(operation_type="create_collection").inc()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {str(e)}")
            return False
        finally:
            VECTOR_STORE_LATENCY.labels(operation_type="create_collection").observe(
                time.time() - start_time
            )

    async def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection and all its data.

        Args:
            collection_name: Name of the collection to delete

        Returns:
            True if collection was deleted successfully
        """
        start_time = time.time()
        try:
            self.client.delete_collection(collection_name=collection_name)
            logger.info(f"Deleted collection {collection_name}")
            VECTOR_STORE_OPERATIONS.labels(operation_type="delete_collection").inc()
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {str(e)}")
            return False
        finally:
            VECTOR_STORE_LATENCY.labels(operation_type="delete_collection").observe(
                time.time() - start_time
            )

    async def list_collections(self) -> List[str]:
        """
        List all available collections.

        Returns:
            List of collection names
        """
        start_time = time.time()
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            VECTOR_STORE_OPERATIONS.labels(operation_type="list_collections").inc()
            return collection_names
        except Exception as e:
            logger.error(f"Failed to list collections: {str(e)}")
            return []
        finally:
            VECTOR_STORE_LATENCY.labels(operation_type="list_collections").observe(
                time.time() - start_time
            )

    async def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Dictionary with collection information or None if not found
        """
        start_time = time.time()
        try:
            info = self.client.get_collection(collection_name=collection_name)
            VECTOR_STORE_OPERATIONS.labels(operation_type="get_collection_info").inc()
            return {
                "name": info.name,
                "status": info.status,
                "vector_size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance,
                "vector_count": info.vectors_count,
                "segments_count": info.segments_count,
                "on_disk_payload": info.config.params.vectors.on_disk,
            }
        except Exception as e:
            logger.error(f"Failed to get info for collection {collection_name}: {str(e)}")
            return None
        finally:
            VECTOR_STORE_LATENCY.labels(operation_type="get_collection_info").observe(
                time.time() - start_time
            )

    async def insert_vectors(
        self,
        collection_name: str,
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> bool:
        """
        Insert vectors with metadata into a collection.

        Args:
            collection_name: Name of the collection
            vectors: List of vectors
            metadata: List of metadata dictionaries
            ids: Optional list of IDs for the vectors

        Returns:
            True if vectors were inserted successfully
        """
        start_time = time.time()
        
        if len(vectors) != len(metadata):
            logger.error("Vectors and metadata lists must have the same length")
            return False
            
        try:
            # Generate random IDs if not provided
            if ids is None:
                from uuid import uuid4
                ids = [str(uuid4()) for _ in range(len(vectors))]
            
            # Convert string IDs to UUID objects if needed
            points = [
                models.PointStruct(
                    id=id_val,
                    vector=vector,
                    payload=meta,
                )
                for id_val, vector, meta in zip(ids, vectors, metadata)
            ]
            
            # Upsert vectors
            self.client.upsert(
                collection_name=collection_name,
                points=points,
            )
            
            logger.info(f"Inserted {len(vectors)} vectors into {collection_name}")
            VECTOR_STORE_OPERATIONS.labels(operation_type="insert_vectors").inc()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert vectors into {collection_name}: {str(e)}")
            return False
        finally:
            VECTOR_STORE_LATENCY.labels(operation_type="insert_vectors").observe(
                time.time() - start_time
            )

    async def search_vectors(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 5,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in a collection.

        Args:
            collection_name: Name of the collection
            query_vector: Query vector
            top_k: Number of closest vectors to return
            filter_conditions: Optional filter conditions

        Returns:
            List of search results with vectors, metadata, and scores
        """
        start_time = time.time()
        try:
            # Convert filter conditions to Qdrant filter
            filter_obj = None
            if filter_conditions:
                filter_obj = self._build_filter(filter_conditions)
            
            # Perform search
            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=filter_obj,
                with_vectors=True,
            )
            
            # Format results
            results = []
            for hit in search_results:
                results.append({
                    "id": hit.id,
                    "score": hit.score,
                    "vector": hit.vector,
                    "payload": hit.payload,
                })
            
            logger.info(f"Found {len(results)} results in {collection_name}")
            VECTOR_STORE_OPERATIONS.labels(operation_type="search_vectors").inc()
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search in {collection_name}: {str(e)}")
            return []
        finally:
            VECTOR_STORE_LATENCY.labels(operation_type="search_vectors").observe(
                time.time() - start_time
            )

    async def delete_vectors(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Delete vectors from a collection by IDs or filter.

        Args:
            collection_name: Name of the collection
            ids: Optional list of vector IDs to delete
            filter_conditions: Optional filter to select vectors for deletion

        Returns:
            True if vectors were deleted successfully
        """
        start_time = time.time()
        try:
            if ids is not None:
                # Delete by IDs
                self.client.delete(
                    collection_name=collection_name,
                    points_selector=models.PointIdsList(
                        points=ids,
                    ),
                )
                logger.info(f"Deleted {len(ids)} vectors from {collection_name}")
            elif filter_conditions is not None:
                # Delete by filter
                filter_obj = self._build_filter(filter_conditions)
                self.client.delete(
                    collection_name=collection_name,
                    points_selector=models.FilterSelector(
                        filter=filter_obj,
                    ),
                )
                logger.info(f"Deleted vectors from {collection_name} using filter")
            else:
                logger.error("Either ids or filter_conditions must be provided")
                return False
                
            VECTOR_STORE_OPERATIONS.labels(operation_type="delete_vectors").inc()
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete vectors from {collection_name}: {str(e)}")
            return False
        finally:
            VECTOR_STORE_LATENCY.labels(operation_type="delete_vectors").observe(
                time.time() - start_time
            )

    async def get_vectors(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve vectors from a collection by IDs or filter.

        Args:
            collection_name: Name of the collection
            ids: Optional list of vector IDs to retrieve
            filter_conditions: Optional filter to select vectors
            limit: Maximum number of vectors to return
            offset: Offset for pagination

        Returns:
            List of vectors with metadata
        """
        start_time = time.time()
        try:
            if ids is not None:
                # Retrieve by IDs
                response = self.client.retrieve(
                    collection_name=collection_name,
                    ids=ids,
                    with_vectors=True,
                )
                results = []
                for point in response:
                    results.append({
                        "id": point.id,
                        "vector": point.vector,
                        "payload": point.payload,
                    })
            elif filter_conditions is not None:
                # Retrieve by filter with scrolling
                filter_obj = self._build_filter(filter_conditions)
                response = self.client.scroll(
                    collection_name=collection_name,
                    scroll_filter=filter_obj,
                    limit=limit,
                    offset=offset,
                    with_vectors=True,
                )
                results = []
                for point in response[0]:  # First element is points, second is next_page_offset
                    results.append({
                        "id": point.id,
                        "vector": point.vector,
                        "payload": point.payload,
                    })
            else:
                # Retrieve all with scrolling (paginated)
                response = self.client.scroll(
                    collection_name=collection_name,
                    limit=limit,
                    offset=offset,
                    with_vectors=True,
                )
                results = []
                for point in response[0]:  # First element is points, second is next_page_offset
                    results.append({
                        "id": point.id,
                        "vector": point.vector,
                        "payload": point.payload,
                    })
                    
            logger.info(f"Retrieved {len(results)} vectors from {collection_name}")
            VECTOR_STORE_OPERATIONS.labels(operation_type="get_vectors").inc()
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get vectors from {collection_name}: {str(e)}")
            return []
        finally:
            VECTOR_STORE_LATENCY.labels(operation_type="get_vectors").observe(
                time.time() - start_time
            )

    def _build_filter(self, filter_conditions: Dict[str, Any]) -> models.Filter:
        """
        Build a Qdrant filter from dictionary conditions.

        Args:
            filter_conditions: Dictionary with filter conditions

        Returns:
            Qdrant filter object
        """
        # This is a simplified implementation that handles basic filters
        # For more complex filters, this would need to be expanded
        
        conditions = []
        
        for field, condition in filter_conditions.items():
            if isinstance(condition, dict):
                # Handle operators like $gt, $lt, etc.
                for op, value in condition.items():
                    if op == "$eq" or op == "==":
                        conditions.append(models.FieldCondition(
                            key=field,
                            match=models.MatchValue(value=value),
                        ))
                    elif op == "$gt" or op == ">":
                        conditions.append(models.FieldCondition(
                            key=field,
                            range=models.Range(gt=value),
                        ))
                    elif op == "$gte" or op == ">=":
                        conditions.append(models.FieldCondition(
                            key=field,
                            range=models.Range(gte=value),
                        ))
                    elif op == "$lt" or op == "<":
                        conditions.append(models.FieldCondition(
                            key=field,
                            range=models.Range(lt=value),
                        ))
                    elif op == "$lte" or op == "<=":
                        conditions.append(models.FieldCondition(
                            key=field,
                            range=models.Range(lte=value),
                        ))
                    elif op == "$in":
                        conditions.append(models.FieldCondition(
                            key=field,
                            match=models.MatchAny(any=value),
                        ))
            else:
                # Simple equality
                conditions.append(models.FieldCondition(
                    key=field,
                    match=models.MatchValue(value=condition),
                ))
        
        if len(conditions) == 1:
            return models.Filter(must=[conditions[0]])
        else:
            return models.Filter(must=conditions)


async def demo():
    """Run a simple demonstration of the vector store capabilities."""
    # Create a vector store instance
    vector_store = VectorStore()
    
    # Collection name for the demo
    collection_name = "demo_collection"
    
    # Create a collection (if it doesn't exist)
    await vector_store.create_collection(
        collection_name=collection_name,
        vector_size=4,  # Using small vectors for demo
    )
    
    # List available collections
    collections = await vector_store.list_collections()
    print(f"Available collections: {collections}")
    
    # Get collection info
    info = await vector_store.get_collection_info(collection_name)
    print(f"Collection info: {json.dumps(info, indent=2)}")
    
    # Insert sample vectors
    vectors = [
        [1.0, 0.0, 0.0, 0.0],  # Vector 1
        [0.0, 1.0, 0.0, 0.0],  # Vector 2
        [0.0, 0.0, 1.0, 0.0],  # Vector 3
        [0.0, 0.0, 0.0, 1.0],  # Vector 4
        [0.5, 0.5, 0.5, 0.5],  # Vector 5
    ]
    
    metadata = [
        {"name": "Vector 1", "category": "red", "value": 10},
        {"name": "Vector 2", "category": "blue", "value": 20},
        {"name": "Vector 3", "category": "green", "value": 30},
        {"name": "Vector 4", "category": "yellow", "value": 40},
        {"name": "Vector 5", "category": "mixed", "value": 50},
    ]
    
    success = await vector_store.insert_vectors(
        collection_name=collection_name,
        vectors=vectors,
        metadata=metadata,
    )
    print(f"Insert success: {success}")
    
    # Search for similar vectors
    query = [0.9, 0.1, 0.0, 0.0]  # Similar to Vector 1
    results = await vector_store.search_vectors(
        collection_name=collection_name,
        query_vector=query,
        top_k=3,
    )
    print("\nSearch results:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['payload']['name']} (score: {result['score']:.4f})")
    
    # Search with filter
    filter_results = await vector_store.search_vectors(
        collection_name=collection_name,
        query_vector=query,
        top_k=3,
        filter_conditions={"category": "blue"},
    )
    print("\nFiltered search results (blue only):")
    for i, result in enumerate(filter_results):
        print(f"{i+1}. {result['payload']['name']} (score: {result['score']:.4f})")
    
    # Get vectors with filter
    high_value_vectors = await vector_store.get_vectors(
        collection_name=collection_name,
        filter_conditions={"value": {"$gte": 30}},
    )
    print("\nHigh value vectors (value >= 30):")
    for i, vector in enumerate(high_value_vectors):
        print(f"{i+1}. {vector['payload']['name']} (value: {vector['payload']['value']})")
    
    # Clean up (optional)
    # Uncomment the line below to delete the demo collection
    # await vector_store.delete_collection(collection_name)


if __name__ == "__main__":
    asyncio.run(demo())
