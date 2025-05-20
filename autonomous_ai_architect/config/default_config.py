"""
Default configuration for the Autonomous AI Architect system.

This module provides default configuration values that can be overridden
by environment variables or a configuration file.
"""

from typing import Dict, Any, Optional, List

from pydantic import BaseModel, Field

class OpenAIConfig(BaseModel):
    """Configuration for OpenAI services."""
    api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key. If None, will try to use environment variable."
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Model to use for text embeddings"
    )
    completion_model: str = Field(
        default="gpt-4-turbo",
        description="Model to use for text completions"
    )
    api_base: Optional[str] = Field(
        default=None,
        description="Base URL for API requests. Use default OpenAI API if None."
    )

class QdrantConfig(BaseModel):
    """Configuration for Qdrant vector database."""
    host: str = Field(
        default="localhost",
        description="Qdrant server host"
    )
    port: int = Field(
        default=6333,
        description="Qdrant server HTTP port"
    )
    grpc_port: int = Field(
        default=6334,
        description="Qdrant server gRPC port"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Qdrant API key for authentication"
    )
    https: bool = Field(
        default=False,
        description="Whether to use HTTPS for Qdrant connection"
    )

class RedisConfig(BaseModel):
    """Configuration for Redis."""
    url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    password: Optional[str] = Field(
        default=None,
        description="Redis password"
    )
    ssl: bool = Field(
        default=False,
        description="Whether to use SSL for Redis connection"
    )

class KnowledgeBaseConfig(BaseModel):
    """Configuration for knowledge base."""
    collection_name: str = Field(
        default="ai_architect_knowledge",
        description="Name of the Qdrant collection to use for knowledge storage"
    )
    chunk_size: int = Field(
        default=1000,
        description="Size of text chunks for storage"
    )
    chunk_overlap: int = Field(
        default=200,
        description="Overlap between consecutive chunks"
    )

class InfraConfig(BaseModel):
    """Configuration for infrastructure components."""
    metrics_port: int = Field(
        default=8000,
        description="Port for Prometheus metrics server"
    )
    log_dir: str = Field(
        default="./logs",
        description="Directory for log files"
    )

class AgentsConfig(BaseModel):
    """Configuration for agent system."""
    max_retries: int = Field(
        default=3,
        description="Maximum number of retries for failed tasks"
    )
    retry_delay: int = Field(
        default=5,
        description="Delay between retries in seconds"
    )
    stop_on_failure: bool = Field(
        default=True,
        description="Whether to stop workflow execution on task failure"
    )
    default_timeout: int = Field(
        default=300,
        description="Default timeout for task execution in seconds"
    )

class Config(BaseModel):
    """Main configuration."""
    environment: str = Field(
        default="development",
        description="Environment (development, staging, production)"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    openai: OpenAIConfig = Field(
        default_factory=OpenAIConfig,
        description="OpenAI configuration"
    )
    qdrant: QdrantConfig = Field(
        default_factory=QdrantConfig,
        description="Qdrant configuration"
    )
    redis: RedisConfig = Field(
        default_factory=RedisConfig,
        description="Redis configuration"
    )
    knowledge_base: KnowledgeBaseConfig = Field(
        default_factory=KnowledgeBaseConfig,
        description="Knowledge base configuration"
    )
    infra: InfraConfig = Field(
        default_factory=InfraConfig,
        description="Infrastructure configuration"
    )
    agents: AgentsConfig = Field(
        default_factory=AgentsConfig,
        description="Agents configuration"
    )

def get_config() -> Config:
    """
    Get configuration from environment variables and config file.
    
    Returns:
        Config object
    """
    # This is a simplified implementation - in a real application,
    # you would load configuration from environment variables and/or
    # a configuration file.
    return Config()
