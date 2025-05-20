#!/usr/bin/env python3
"""
Configuration Manager Module

This module provides a centralized way to manage configuration settings
and secrets for the Autonomous AI Architect system.

Key features:
- Load configuration from environment variables, config files, and defaults
- Secure handling of sensitive information
- Configuration validation
- Support for different environments (dev, test, prod)
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class LLMConfig(BaseModel):
    """Configuration settings for LLM providers."""
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")
    azure_api_key: Optional[str] = Field(None, description="Azure OpenAI API key")
    azure_endpoint: Optional[str] = Field(None, description="Azure OpenAI endpoint URL")
    default_model: str = Field("gpt-4", description="Default model to use")
    embedding_model: str = Field("text-embedding-3-small", description="Model to use for embeddings")
    request_timeout: int = Field(60, description="Request timeout in seconds")
    temperature: float = Field(0.7, description="LLM temperature setting")
    max_tokens: int = Field(2000, description="Maximum number of tokens per request")

class InfraConfig(BaseModel):
    """Infrastructure configuration settings."""
    redis_url: str = Field("redis://localhost:6379/0", description="Redis connection URL")
    metrics_port: int = Field(8000, description="Prometheus metrics port")
    docker_socket_path: str = Field("/var/run/docker.sock", description="Docker socket path")
    sandbox_container_name: str = Field("openinterpreter_sandbox", description="Sandbox container name")

class QdrantConfig(BaseModel):
    """Configuration for Qdrant vector database."""
    host: str = Field("localhost", description="Qdrant server host")
    port: int = Field(6333, description="Qdrant server HTTP port")
    grpc_port: int = Field(6334, description="Qdrant server gRPC port")
    api_key: Optional[str] = Field(None, description="Qdrant API key for authentication")
    https: bool = Field(False, description="Whether to use HTTPS for Qdrant connection")

class KnowledgeBaseConfig(BaseModel):
    """Configuration for knowledge base."""
    collection_name: str = Field("ai_architect_knowledge", description="Name of the Qdrant collection")
    chunk_size: int = Field(1000, description="Size of text chunks for storage")
    chunk_overlap: int = Field(200, description="Overlap between consecutive chunks")

class AgentsConfig(BaseModel):
    """Configuration for agent system."""
    max_retries: int = Field(3, description="Maximum number of retries for failed tasks")
    retry_delay: int = Field(5, description="Delay between retries in seconds")
    stop_on_failure: bool = Field(True, description="Whether to stop workflow execution on task failure")
    default_timeout: int = Field(300, description="Default timeout for task execution in seconds")

class Config(BaseModel):
    """Main configuration container."""
    environment: str = Field("development", description="Environment (development, test, production)")
    debug: bool = Field(False, description="Enable debug mode")
    log_level: str = Field("INFO", description="Logging level")
    llm: LLMConfig = Field(default_factory=LLMConfig, description="LLM configuration")
    infra: InfraConfig = Field(default_factory=InfraConfig, description="Infrastructure configuration")
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig, description="Qdrant configuration")
    knowledge_base: KnowledgeBaseConfig = Field(default_factory=KnowledgeBaseConfig, description="Knowledge base configuration")
    agents: AgentsConfig = Field(default_factory=AgentsConfig, description="Agents configuration")
    project_root: Optional[str] = Field(None, description="Project root directory")

class ConfigManager:
    """
    Configuration manager for the Autonomous AI Architect system.
    
    Handles loading configuration from various sources and providing
    a validated configuration object.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Optional path to a JSON configuration file
        """
        self.config = Config()
        self._config_path = Path(config_path) if config_path else None
        self._loaded = False
    
    def load_config(self) -> Config:
        """
        Load configuration from all sources and return a validated Config object.
        
        The priority order is:
        1. Environment variables
        2. Configuration file
        3. Default values
        
        Returns:
            A validated Config object
        """
        if self._loaded:
            return self.config
        
        # Set project root if not already set
        if not self.config.project_root:
            # Try to find the project root (assumes this module is in <project_root>/core)
            current_dir = Path(__file__).resolve().parent
            if current_dir.name == "core":
                self.config.project_root = str(current_dir.parent)
        
        # Load from config file if available
        self._load_from_file()
        
        # Load from environment variables (overrides file config)
        self._load_from_env()
        
        # Validate the configuration
        self.config = Config(**self.config.dict())
        
        self._loaded = True
        return self.config
    
    def _load_from_file(self) -> None:
        """Load configuration from a JSON file if available."""
        if not self._config_path:
            # Look for config in default locations
            potential_paths = [
                Path.cwd() / "config.json",
                Path.home() / ".config" / "autonomous_ai_architect" / "config.json",
                Path("/etc/autonomous_ai_architect/config.json")
            ]
            
            # If project_root is set, also check there
            if self.config.project_root:
                potential_paths.insert(0, Path(self.config.project_root) / "config.json")
            
            for path in potential_paths:
                if path.exists():
                    self._config_path = path
                    break
        
        if self._config_path and self._config_path.exists():
            try:
                with open(self._config_path, 'r') as f:
                    config_data = json.load(f)
                
                # Update configuration from file
                self._update_config_recursive(config_data)
                logger.info(f"Loaded configuration from {self._config_path}")
            
            except Exception as e:
                logger.error(f"Error loading configuration from {self._config_path}: {str(e)}")
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # Map of environment variable names to config fields
        env_mappings = {
            "AAA_ENV": "environment",
            "AAA_DEBUG": ("debug", lambda x: x.lower() == "true"),
            "AAA_LOG_LEVEL": "log_level",
            "AAA_OPENAI_API_KEY": "llm.openai_api_key",
            "AAA_AZURE_API_KEY": "llm.azure_api_key",
            "AAA_AZURE_ENDPOINT": "llm.azure_endpoint",
            "AAA_DEFAULT_MODEL": "llm.default_model",
            "AAA_REQUEST_TIMEOUT": ("llm.request_timeout", int),
            "AAA_TEMPERATURE": ("llm.temperature", float),
            "AAA_MAX_TOKENS": ("llm.max_tokens", int),
            "AAA_REDIS_URL": "infra.redis_url",
            "AAA_METRICS_PORT": ("infra.metrics_port", int),
            "AAA_DOCKER_SOCKET": "infra.docker_socket_path",
            "AAA_SANDBOX_CONTAINER": "infra.sandbox_container_name",
        }
        
        for env_var, config_field in env_mappings.items():
            # Check if the environment variable is set
            if env_var in os.environ:
                value = os.environ[env_var]
                
                # If config_field is a tuple, it includes a conversion function
                if isinstance(config_field, tuple):
                    field_path, converter = config_field
                    try:
                        value = converter(value)
                    except Exception as e:
                        logger.error(f"Error converting {env_var} value '{value}': {str(e)}")
                        continue
                else:
                    field_path = config_field
                
                # Set the value in the config
                self._set_nested_attr(field_path, value)
    
    def _update_config_recursive(self, config_data: Dict[str, Any], prefix: str = "") -> None:
        """
        Update configuration values recursively from a dictionary.
        
        Args:
            config_data: Dictionary containing configuration values
            prefix: Prefix for nested keys
        """
        for key, value in config_data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                # Recurse into nested dictionaries
                self._update_config_recursive(value, full_key)
            else:
                # Set the value directly
                self._set_nested_attr(full_key, value)
    
    def _set_nested_attr(self, attr_path: str, value: Any) -> None:
        """
        Set a nested attribute in the config object.
        
        Args:
            attr_path: Dot-separated path to the attribute
            value: Value to set
        """
        parts = attr_path.split('.')
        current = self.config.dict()
        
        # Navigate to the inner object that contains the field
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                logger.warning(f"Unknown configuration path: {attr_path}")
                return
            current = current[part]
            
            # If we've navigated past a scalar, we can't keep going
            if not isinstance(current, dict):
                logger.warning(f"Cannot set nested attribute {attr_path}, {'.'.join(parts[:i+1])} is not a dict")
                return
        
        # Set the final field
        final_key = parts[-1]
        if final_key in current:
            # Create a new dictionary with the updated value
            updated_dict = self.config.dict()
            
            # Navigate to the correct place and update the value
            current_dict = updated_dict
            for part in parts[:-1]:
                current_dict = current_dict[part]
            current_dict[final_key] = value
            
            # Recreate the config with the updated dictionary
            self.config = Config(**updated_dict)
        else:
            logger.warning(f"Unknown configuration field: {attr_path}")

# Global config manager instance
config_manager = ConfigManager()

def get_config() -> Config:
    """
    Get the current configuration.
    
    Returns:
        The current validated configuration
    """
    return config_manager.load_config()
