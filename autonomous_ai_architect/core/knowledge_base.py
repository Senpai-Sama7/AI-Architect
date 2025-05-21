#!/usr/bin/env python3
"""
Knowledge Base Manager

This module provides a comprehensive knowledge base management system that combines
text embedding generation and vector storage. It enables storing, retrieving,
and searching documents or chunks of information using semantic similarity.

Key features:
- Document chunking and embedding
- Semantic search with relevance scoring
- Metadata filtering
- Document versioning
- Batched operations
- Observability with metrics
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union, Set

from prometheus_client import Counter, Histogram

from autonomous_ai_architect.tools.embedding_generator import EmbeddingGenerator
from autonomous_ai_architect.tools.vector_store import VectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Prometheus metrics
KB_OPERATIONS = Counter(
    "knowledge_base_operations_total",
    "Total number of knowledge base operations",
    ["operation_type", "status"]
)
KB_LATENCY = Histogram(
    "knowledge_base_latency_seconds",
    "Knowledge base operation latency in seconds",
    ["operation_type"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30]
)


class KnowledgeBase:
    """
    Comprehensive knowledge base manager with semantic search capabilities.
    """

    def __init__(
        self,
        collection_name: str = "knowledge_base",
        embedding_model: str = "text-embedding-3-small",
        embedding_dimensions: int = 1536,
        vector_store_host: str = "localhost",
        vector_store_port: int = 6333,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        batch_size: int = 16,
    ):
        """
        Initialize the knowledge base.

        Args:
            collection_name: Name of the vector store collection
            embedding_model: Name of the embedding model to use
            embedding_dimensions: Dimensionality of the embeddings
            vector_store_host: Host of the vector database
            vector_store_port: Port of the vector database
            chunk_size: Maximum size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
            batch_size: Batch size for processing
        """
        self.collection_name = collection_name
        self.embedding_dimensions = embedding_dimensions
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator(
            model=embedding_model,
            dimensions=embedding_dimensions,
            batch_size=batch_size,
        )
        
        # Initialize vector store
        self.vector_store = VectorStore(
            host=vector_store_host,
            port=vector_store_port,
        )
        
        logger.info(f"Initialized knowledge base with collection: {collection_name}")

    async def initialize(self) -> bool:
        """
        Initialize the knowledge base, creating the collection if needed.

        Returns:
            True if initialization was successful
        """
        logger.info("Initializing knowledge base")
        start_time = time.time()
        
        try:
            # Create the collection if it doesn't exist
            success = await self.vector_store.create_collection(
                collection_name=self.collection_name,
                vector_size=self.embedding_dimensions,
            )
            
            KB_OPERATIONS.labels(operation_type="initialize", status="success" if success else "error").inc()
            
            return success
        except Exception as e:
            logger.error(f"Failed to initialize knowledge base: {str(e)}")
            KB_OPERATIONS.labels(operation_type="initialize", status="error").inc()
            return False
        finally:
            KB_LATENCY.labels(operation_type="initialize").observe(time.time() - start_time)

    async def add_document(
        self,
        document: str,
        metadata: Dict[str, Any],
        document_id: Optional[str] = None,
        chunk: bool = True,
    ) -> List[str]:
        """
        Add a document to the knowledge base, chunking if needed.

        Args:
            document: The document text
            metadata: Metadata associated with the document
            document_id: Optional unique ID for the document
            chunk: Whether to split the document into chunks

        Returns:
            List of chunk IDs created
        """
        logger.info("Adding document to knowledge base")
        start_time = time.time()
        
        try:
            # Generate document ID if not provided
            if document_id is None:
                document_id = self._generate_id(document)
            
            # Add document_id to metadata
            metadata = {**metadata, "document_id": document_id}
            
            # Add timestamp if not present
            if "timestamp" not in metadata:
                metadata["timestamp"] = datetime.now().isoformat()
            
            # Split document into chunks if needed
            if chunk and len(document) > self.chunk_size:
                chunks = self._chunk_text(document)
                logger.info(f"Split document into {len(chunks)} chunks")
            else:
                chunks = [document]
            
            # Generate embeddings for all chunks
            embeddings = await self.embedding_generator.get_embeddings(chunks)
            
            # Create metadata for each chunk
            chunk_ids = []
            chunk_metadatas = []
            
            for i, chunk_text in enumerate(chunks):
                chunk_id = f"{document_id}_{i}" if len(chunks) > 1 else document_id
                chunk_metadata = {
                    **metadata,
                    "chunk_id": chunk_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "text": chunk_text,  # Store the text in metadata for retrieval
                }
                chunk_ids.append(chunk_id)
                chunk_metadatas.append(chunk_metadata)
            
            # Insert vectors into the vector store
            success = await self.vector_store.insert_vectors(
                collection_name=self.collection_name,
                vectors=embeddings,
                metadata=chunk_metadatas,
                ids=chunk_ids,
            )
            
            if success:
                logger.info(f"Added document with ID {document_id} ({len(chunks)} chunks)")
                KB_OPERATIONS.labels(operation_type="add_document", status="success").inc()
                return chunk_ids
            else:
                logger.error(f"Failed to add document with ID {document_id}")
                KB_OPERATIONS.labels(operation_type="add_document", status="error").inc()
                return []
                
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            KB_OPERATIONS.labels(operation_type="add_document", status="error").inc()
            return []
        finally:
            KB_LATENCY.labels(operation_type="add_document").observe(time.time() - start_time)

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filter_conditions: Optional[Dict[str, Any]] = None,
        include_text: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for relevant information.

        Args:
            query: The search query
            top_k: Number of results to return
            filter_conditions: Optional filter conditions
            include_text: Whether to include the text in the results

        Returns:
            List of search results with relevance scores
        """
        logger.info("Searching knowledge base")
        start_time = time.time()
        
        try:
            # Generate embedding for the query
            query_embedding = await self.embedding_generator.get_embeddings([query])
            
            if not query_embedding:
                logger.error("Failed to generate embedding for query")
                KB_OPERATIONS.labels(operation_type="search", status="error").inc()
                return []
            
            # Search the vector store
            results = await self.vector_store.search_vectors(
                collection_name=self.collection_name,
                query_vector=query_embedding[0],
                top_k=top_k,
                filter_conditions=filter_conditions,
            )
            
            # Format results
            formatted_results = []
            for result in results:
                entry = {
                    "id": result["id"],
                    "score": result["score"],
                    "metadata": {k: v for k, v in result["payload"].items() if k != "text" or include_text},
                }
                formatted_results.append(entry)
            
            logger.info(f"Search returned {len(formatted_results)} results")
            KB_OPERATIONS.labels(operation_type="search", status="success").inc()
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching knowledge base: {str(e)}")
            KB_OPERATIONS.labels(operation_type="search", status="error").inc()
            return []
        finally:
            KB_LATENCY.labels(operation_type="search").observe(time.time() - start_time)

    async def get_document(
        self,
        document_id: str,
        include_text: bool = True,
    ) -> Dict[str, Any]:
        """
        Retrieve a document by ID, reassembling chunks if needed.

        Args:
            document_id: ID of the document to retrieve
            include_text: Whether to include the text in the result

        Returns:
            Document with metadata and text (if include_text is True)
        """
        logger.info(f"Retrieving document with ID {document_id}")
        start_time = time.time()
        
        try:
            # Get all chunks for this document
            chunks = await self.vector_store.get_vectors(
                collection_name=self.collection_name,
                filter_conditions={"document_id": document_id},
            )
            
            if not chunks:
                logger.warning(f"Document with ID {document_id} not found")
                KB_OPERATIONS.labels(operation_type="get_document", status="not_found").inc()
                return {}
            
            # Sort chunks by index
            chunks.sort(key=lambda x: x["payload"].get("chunk_index", 0))
            
            # Combine metadata from the first chunk (excluding chunk-specific fields)
            metadata = {k: v for k, v in chunks[0]["payload"].items() 
                      if k not in ["chunk_id", "chunk_index", "total_chunks", "text"]}
            
            # Combine text if requested
            if include_text:
                text = "".join(chunk["payload"].get("text", "") for chunk in chunks)
                metadata["text"] = text
            
            result = {
                "document_id": document_id,
                "metadata": metadata,
                "chunk_count": len(chunks),
            }
            
            logger.info(f"Retrieved document with ID {document_id} ({len(chunks)} chunks)")
            KB_OPERATIONS.labels(operation_type="get_document", status="success").inc()
            
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving document {document_id}: {str(e)}")
            KB_OPERATIONS.labels(operation_type="get_document", status="error").inc()
            return {}
        finally:
            KB_LATENCY.labels(operation_type="get_document").observe(time.time() - start_time)

    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and all its chunks from the knowledge base.

        Args:
            document_id: ID of the document to delete

        Returns:
            True if deletion was successful
        """
        logger.info(f"Deleting document with ID {document_id}")
        start_time = time.time()
        
        try:
            # Delete all vectors with this document_id
            success = await self.vector_store.delete_vectors(
                collection_name=self.collection_name,
                filter_conditions={"document_id": document_id},
            )
            
            logger.info(f"Deleted document with ID {document_id}")
            KB_OPERATIONS.labels(
                operation_type="delete_document", 
                status="success" if success else "error"
            ).inc()
            
            return success
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            KB_OPERATIONS.labels(operation_type="delete_document", status="error").inc()
            return False
        finally:
            KB_LATENCY.labels(operation_type="delete_document").observe(time.time() - start_time)

    async def update_document(
        self,
        document_id: str,
        new_document: str,
        new_metadata: Optional[Dict[str, Any]] = None,
        chunk: bool = True,
    ) -> List[str]:
        """
        Update an existing document in the knowledge base.

        Args:
            document_id: ID of the document to update
            new_document: New document text
            new_metadata: New metadata (will be merged with existing)
            chunk: Whether to split the document into chunks

        Returns:
            List of chunk IDs created
        """
        logger.info(f"Updating document with ID {document_id}")
        start_time = time.time()
        
        try:
            # Get existing document to merge metadata
            existing = await self.get_document(document_id, include_text=False)
            
            # Prepare merged metadata
            if existing and "metadata" in existing:
                if new_metadata:
                    # Merge existing and new metadata
                    metadata = {**existing["metadata"], **new_metadata}
                else:
                    # Use existing metadata
                    metadata = existing["metadata"]
            else:
                # Use new metadata or empty dict
                metadata = new_metadata or {}
            
            # Add/update version information
            if "version" in metadata:
                metadata["version"] = metadata["version"] + 1
                metadata["previous_version"] = metadata["version"] - 1
            else:
                metadata["version"] = 1
            
            # Add update timestamp
            metadata["updated_at"] = datetime.now().isoformat()
            
            # Delete existing document
            await self.delete_document(document_id)
            
            # Add the updated document
            chunk_ids = await self.add_document(
                document=new_document,
                metadata=metadata,
                document_id=document_id,
                chunk=chunk,
            )
            
            logger.info(f"Updated document with ID {document_id}")
            KB_OPERATIONS.labels(
                operation_type="update_document", 
                status="success" if chunk_ids else "error"
            ).inc()
            
            return chunk_ids
            
        except Exception as e:
            logger.error(f"Error updating document {document_id}: {str(e)}")
            KB_OPERATIONS.labels(operation_type="update_document", status="error").inc()
            return []
        finally:
            KB_LATENCY.labels(operation_type="update_document").observe(time.time() - start_time)

    async def list_documents(
        self,
        filter_conditions: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        List documents in the knowledge base.

        Args:
            filter_conditions: Optional filter conditions
            limit: Maximum number of documents to return
            offset: Offset for pagination

        Returns:
            List of documents with metadata
        """
        logger.info("Listing documents in knowledge base")
        start_time = time.time()
        
        try:
            # Get raw vectors with filter
            raw_results = await self.vector_store.get_vectors(
                collection_name=self.collection_name,
                filter_conditions=filter_conditions,
                limit=limit,
                offset=offset,
            )
            
            # Process results to get unique documents
            document_ids = set()
            documents = []
            
            for result in raw_results:
                document_id = result["payload"].get("document_id")
                
                if document_id and document_id not in document_ids:
                    document_ids.add(document_id)
                    
                    # Extract metadata (excluding chunk-specific and text fields)
                    metadata = {k: v for k, v in result["payload"].items() 
                               if k not in ["chunk_id", "chunk_index", "total_chunks", "text"]}
                    
                    documents.append({
                        "document_id": document_id,
                        "metadata": metadata,
                    })
            
            logger.info(f"Listed {len(documents)} documents")
            KB_OPERATIONS.labels(operation_type="list_documents", status="success").inc()
            
            return documents
            
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            KB_OPERATIONS.labels(operation_type="list_documents", status="error").inc()
            return []
        finally:
            KB_LATENCY.labels(operation_type="list_documents").observe(time.time() - start_time)

    async def add_documents_batch(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        document_ids: Optional[List[str]] = None,
        chunk: bool = True,
    ) -> List[str]:
        """
        Add multiple documents to the knowledge base in batch.

        Args:
            documents: List of document texts
            metadatas: List of metadata dictionaries
            document_ids: Optional list of document IDs
            chunk: Whether to split documents into chunks

        Returns:
            List of document IDs added
        """
        logger.info("Adding documents in batch to knowledge base")
        start_time = time.time()
        
        if len(documents) != len(metadatas):
            logger.error("Number of documents and metadatas must match")
            KB_OPERATIONS.labels(operation_type="add_documents_batch", status="error").inc()
            return []
        
        # Generate document IDs if not provided
        if document_ids is None:
            document_ids = [self._generate_id(doc) for doc in documents]
        elif len(document_ids) != len(documents):
            logger.error("Number of document_ids must match documents if provided")
            KB_OPERATIONS.labels(operation_type="add_documents_batch", status="error").inc()
            return []
        
        try:
            added_ids = []
            
            # Process in batches
            for i in range(0, len(documents), self.batch_size):
                batch_docs = documents[i:i + self.batch_size]
                batch_metas = metadatas[i:i + self.batch_size]
                batch_ids = document_ids[i:i + self.batch_size]
                
                # Process each document in the batch
                tasks = [
                    self.add_document(
                        document=doc,
                        metadata=meta,
                        document_id=doc_id,
                        chunk=chunk
                    )
                    for doc, meta, doc_id in zip(batch_docs, batch_metas, batch_ids)
                ]
                
                # Wait for all tasks to complete
                results = await asyncio.gather(*tasks)
                
                # Collect document IDs that were successfully added
                for doc_id, chunk_ids in zip(batch_ids, results):
                    if chunk_ids:
                        added_ids.append(doc_id)
            
            logger.info(f"Added {len(added_ids)} documents in batch")
            KB_OPERATIONS.labels(operation_type="add_documents_batch", status="success").inc()
            
            return added_ids
            
        except Exception as e:
            logger.error(f"Error adding documents in batch: {str(e)}")
            KB_OPERATIONS.labels(operation_type="add_documents_batch", status="error").inc()
            return []
        finally:
            KB_LATENCY.labels(operation_type="add_documents_batch").observe(time.time() - start_time)

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap.

        Args:
            text: The text to split

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end of current chunk
            end = min(start + self.chunk_size, len(text))
            
            # If we're not at the end of the text and not at a whitespace, extend to next whitespace
            if end < len(text) and not text[end].isspace():
                # Look for next whitespace
                next_space = text.find(" ", end)
                if (next_space != -1 and next_space - end < 50):  # Don't extend too far
                    end = next_space
            
            # Extract the chunk
            chunk = text[start:end].strip()
            chunks.append(chunk)
            
            # Move start position for next chunk, accounting for overlap
            start = start + self.chunk_size - self.chunk_overlap
            
            # Make sure we're making progress
            if start <= 0:
                start = end
        
        return chunks

    def _generate_id(self, text: str) -> str:
        """
        Generate a unique ID for a document.

        Args:
            text: The document text

        Returns:
            Unique document ID
        """
        # Use a hash of the text and timestamp for uniqueness
        timestamp = str(time.time())
        hash_input = (text[:1000] + timestamp).encode()  # Use first 1000 chars to avoid very long texts
        return hashlib.md5(hash_input).hexdigest()


async def demo():
    """Run a demonstration of the knowledge base functionality."""
    # Initialize knowledge base
    kb = KnowledgeBase(
        collection_name="demo_kb",
        embedding_model="text-embedding-3-small",
    )
    
    # Initialize the knowledge base
    initialized = await kb.initialize()
    print(f"Knowledge base initialized: {initialized}")
    
    # Sample documents
    documents = [
        """
        Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, 
        especially computer systems. These processes include learning (the acquisition of information 
        and rules for using the information), reasoning (using rules to reach approximate or definite 
        conclusions), and self-correction. Particular applications of AI include expert systems, 
        speech recognition, and machine vision.
        """,
        
        """
        Machine Learning (ML) is a subset of artificial intelligence (AI) that provides systems 
        the ability to automatically learn and improve from experience without being explicitly 
        programmed. ML focuses on the development of computer programs that can access data and 
        use it to learn for themselves. The learning process begins with observations or data, 
        such as examples, direct experience, or instruction, in order to look for patterns in 
        data and make better decisions in the future based on the examples that we provide.
        """,
        
        """
        Deep Learning is a subset of machine learning that employs artificial neural networks 
        with multiple layers (hence "deep"). These networks are designed to mimic the human brain's 
        structure and function, enabling them to "learn" from large amounts of data. Deep learning 
        drives many artificial intelligence applications and services that improve automation, 
        performing analytical and physical tasks without human intervention.
        """,
    ]
    
    # Add documents with metadata
    doc_ids = []
    for i, doc in enumerate(documents):
        metadata = {
            "title": f"AI Document {i+1}",
            "topic": ["AI", "Machine Learning", "Deep Learning"][i],
            "author": "Knowledge Base Demo",
            "importance": i + 1,
        }
        
        result = await kb.add_document(
            document=doc,
            metadata=metadata,
        )
        doc_ids.append(result[0] if result else None)
        print(f"Added document {i+1} with ID: {result[0] if result else 'Failed'}")
    
    # Search the knowledge base
    queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "Explain deep learning",
    ]
    
    for query in queries:
        print(f"\nSearching for: '{query}'")
        results = await kb.search(query, top_k=2)
        
        for i, result in enumerate(results):
            print(f"\nResult {i+1} (score: {result['score']:.4f}):")
            print(f"Title: {result['metadata'].get('title')}")
            print(f"Topic: {result['metadata'].get('topic')}")
            
            # Print a snippet of the text if available
            text = result['metadata'].get('text', '')
            if text:
                snippet = text[:150] + '...' if len(text) > 150 else text
                print(f"Snippet: {snippet}")
    
    # Update a document
    if doc_ids[0]:
        updated_doc = """
        Artificial Intelligence (AI) refers to systems or machines that mimic human intelligence 
        to perform tasks and can iteratively improve themselves based on the information they 
        collect. AI manifests in a number of forms, including chatbots, self-driving cars, and 
        intelligent robotic process automation. AI is increasingly a critical component of modern 
        business operations and strategy.
        """
        
        updated_metadata = {
            "title": "Updated AI Document",
            "version_note": "Refined definition"
        }
        
        updated = await kb.update_document(
            document_id=doc_ids[0],
            new_document=updated_doc,
            new_metadata=updated_metadata,
        )
        print(f"\nUpdated document: {updated}")
        
        # Retrieve the updated document
        doc = await kb.get_document(doc_ids[0])
        print(f"\nRetrieved updated document:")
        print(f"Title: {doc['metadata'].get('title')}")
        print(f"Version: {doc['metadata'].get('version')}")
        print(f"Note: {doc['metadata'].get('version_note')}")
    
    # List all documents
    print("\nListing all documents:")
    all_docs = await kb.list_documents()
    for doc in all_docs:
        print(f"- {doc['document_id']}: {doc['metadata'].get('title')}")
    
    # Clean up (uncomment to delete demo collection)
    # await kb.vector_store.delete_collection("demo_kb")


if __name__ == "__main__":
    asyncio.run(demo())
