#!/usr/bin/env python3
"""
Dynamic LLM Router Module

This module provides a dynamic routing mechanism between OpenAI and Azure OpenAI services
with built-in caching, circuit breaker patterns, and Prometheus metrics monitoring.

Key features:
- Round-robin request routing between OpenAI and Azure OpenAI
- Redis-based caching for the last 5 responses per unique prompt
- Circuit breaker to prevent repeated failures
- Prometheus metrics for monitoring
- Exponential backoff retry strategy for transient errors
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Callable, Awaitable

import aiohttp
import openai
import redis
from prometheus_client import Counter, Histogram, start_http_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Redis configuration
REDIS_URL = "redis://localhost:6379/0"
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

# Prometheus metrics
LLM_REQUESTS = Counter(
    "llm_requests_total", 
    "Total number of LLM requests", 
    ["provider"]
)
LLM_FAILURES = Counter(
    "llm_failures_total", 
    "Total number of LLM failures", 
    ["provider", "error_type"]
)
LLM_LATENCY = Histogram(
    "llm_request_latency_seconds", 
    "LLM request latency in seconds", 
    ["provider"], 
    buckets=[0.1, 0.5, 1, 2, 5]
)

# Circuit breaker states
class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, do not use
    HALF_OPEN = "half_open"  # Testing if service is back

# Circuit breaker configuration
@dataclass
class CircuitBreaker:
    failures: int = 0
    state: CircuitState = CircuitState.CLOSED
    last_failure_time: float = 0
    reset_timeout: float = 60.0  # Time in seconds to wait before attempting reset

    def record_failure(self) -> None:
        """Record a failure and potentially trip the circuit breaker."""
        self.failures += 1
        self.last_failure_time = time.time()
        
        if self.failures >= 3 and self.state == CircuitState.CLOSED:
            logger.warning("Circuit breaker tripped")
            self.state = CircuitState.OPEN
    
    def record_success(self) -> None:
        """Record a successful request and reset the failure counter."""
        self.failures = 0
        if self.state != CircuitState.CLOSED:
            logger.info("Circuit breaker reset to CLOSED state")
            self.state = CircuitState.CLOSED
    
    def is_open(self) -> bool:
        """Check if the circuit breaker is open."""
        if self.state == CircuitState.OPEN:
            # Check if we've waited long enough to try again
            if time.time() - self.last_failure_time >= self.reset_timeout:
                logger.info("Circuit breaker transitioning to HALF_OPEN state")
                self.state = CircuitState.HALF_OPEN
                return False
            return True
        return False

# Exception classes
class LLMRouterException(Exception):
    """Base exception for LLM Router."""
    pass

class AllProvidersDownException(LLMRouterException):
    """Exception raised when all LLM providers are unavailable."""
    pass

class TransientErrorException(LLMRouterException):
    """Exception raised for temporary errors that might resolve with retries."""
    pass

class ProviderDownException(LLMRouterException):
    """Exception raised when a specific provider is down."""
    pass

# Global state
request_count = 0
circuit_breakers = {
    "openai": CircuitBreaker(),
    "azure": CircuitBreaker()
}

async def get_cached_response(prompt: str) -> Optional[str]:
    """
    Check if a response for the given prompt exists in the cache.
    
    Args:
        prompt: The input prompt to check in cache
        
    Returns:
        The cached response if available, None otherwise
    """
    # Create a hash of the prompt to use as a key
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    cache_key = f"llm_cache:{prompt_hash}"
    
    # Check if we have a cached response
    if redis_client.exists(cache_key):
        logger.info(f"Cache hit for prompt: {prompt_hash}")
        return redis_client.lindex(cache_key, 0)  # Get the most recent response
    
    return None

async def cache_response(prompt: str, response: str) -> None:
    """
    Store a response in the Redis cache with LRU-like behavior.
    
    Args:
        prompt: The input prompt that generated the response
        response: The LLM response to cache
    """
    # Create a hash of the prompt to use as a key
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    cache_key = f"llm_cache:{prompt_hash}"
    
    # Use Redis LPUSH to add the newest entry at the beginning
    redis_client.lpush(cache_key, response)
    
    # Trim the list to keep only the 5 most recent entries
    redis_client.ltrim(cache_key, 0, 4)
    
    # Set expiry for 1 hour to prevent cache from growing indefinitely
    redis_client.expire(cache_key, 3600)

async def call_openai(prompt: str, **kwargs) -> str:
    """
    Call the OpenAI API with the given prompt.
    
    Args:
        prompt: The input prompt to send to OpenAI
        **kwargs: Additional parameters to pass to the OpenAI API
        
    Returns:
        The text response from OpenAI
    """
    client = openai.AsyncOpenAI()
    
    # Set default parameters if not provided
    model = kwargs.get("model", "gpt-4")
    temperature = kwargs.get("temperature", 0.7)
    max_tokens = kwargs.get("max_tokens", 1000)
    
    start_time = time.time()
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        LLM_LATENCY.labels(provider="openai").observe(time.time() - start_time)
        return response.choices[0].message.content
    except Exception as e:
        error_type = type(e).__name__
        LLM_FAILURES.labels(provider="openai", error_type=error_type).inc()
        
        # Determine if this is a transient error that can be retried
        if isinstance(e, (openai.RateLimitError, aiohttp.ClientError)):
            raise TransientErrorException(f"Transient error from OpenAI: {str(e)}")
        raise LLMRouterException(f"Error from OpenAI: {str(e)}")

async def call_azure(prompt: str, **kwargs) -> str:
    """
    Call the Azure OpenAI API with the given prompt.
    
    Args:
        prompt: The input prompt to send to Azure OpenAI
        **kwargs: Additional parameters to pass to the Azure OpenAI API
        
    Returns:
        The text response from Azure OpenAI
    """
    # In a real scenario, these would be loaded from environment or a secure vault
    from openai import AsyncAzureOpenAI
    
    client = AsyncAzureOpenAI(
        azure_endpoint=kwargs.get("azure_endpoint", "https://your-resource-name.openai.azure.com"),
        api_key=kwargs.get("api_key", "your-api-key"),
        api_version=kwargs.get("api_version", "2023-05-15")
    )
    
    # Set default parameters if not provided
    deployment_name = kwargs.get("deployment_name", "gpt-4")
    temperature = kwargs.get("temperature", 0.7)
    max_tokens = kwargs.get("max_tokens", 1000)
    
    start_time = time.time()
    try:
        response = await client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        LLM_LATENCY.labels(provider="azure").observe(time.time() - start_time)
        return response.choices[0].message.content
    except Exception as e:
        error_type = type(e).__name__
        LLM_FAILURES.labels(provider="azure", error_type=error_type).inc()
        
        # Determine if this is a transient error that can be retried
        if isinstance(e, (aiohttp.ClientError)):
            raise TransientErrorException(f"Transient error from Azure: {str(e)}")
        raise LLMRouterException(f"Error from Azure: {str(e)}")

async def with_retry(
    func: Callable[..., Awaitable[str]], 
    *args: Any, 
    max_retries: int = 3,
    initial_backoff: float = 0.5,
    backoff_factor: float = 2,
    **kwargs: Any
) -> str:
    """
    Execute a function with exponential backoff retry strategy.
    
    Args:
        func: The async function to execute
        *args: Positional arguments to pass to the function
        max_retries: Maximum number of retries (default: 3)
        initial_backoff: Initial backoff time in seconds (default: 0.5)
        backoff_factor: Factor to increase backoff time (default: 2)
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        The result from the function
        
    Raises:
        The last exception encountered if all retries fail
    """
    last_exception = None
    current_backoff = initial_backoff
    
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except TransientErrorException as e:
            last_exception = e
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed with error: {str(e)}")
            logger.info(f"Retrying in {current_backoff} seconds...")
            await asyncio.sleep(current_backoff)
            current_backoff *= backoff_factor
        except Exception as e:
            # Non-transient exceptions should not be retried
            raise e
    
    # If we've exhausted all retries, re-raise the last exception
    raise last_exception

async def route_request(prompt: str, provider: str = "openai", **kwargs) -> str:
    """
    Route an LLM request to the appropriate provider with caching and circuit breaking.
    
    Args:
        prompt: The input prompt to send to the LLM
        provider: The preferred provider (default: "openai")
        **kwargs: Additional parameters to pass to the LLM API
        
    Returns:
        The text response from the LLM
        
    Raises:
        AllProvidersDownException: If all providers are unavailable
    """
    global request_count
    
    # First check cache
    cached_response = await get_cached_response(prompt)
    if cached_response:
        return cached_response
    
    # Determine the order of providers to try
    if provider not in ["openai", "azure"]:
        provider = "openai"  # Default to OpenAI if invalid provider specified
    
    # Use round-robin if no specific provider is requested
    if provider == "openai" or provider == "azure":
        providers = [provider, "azure" if provider == "openai" else "openai"]
    else:
        # Round-robin selection
        request_count += 1
        if request_count % 2 == 0:
            providers = ["openai", "azure"]
        else:
            providers = ["azure", "openai"]
    
    # Try each provider in order
    last_error = None
    for current_provider in providers:
        # Check if this provider's circuit breaker is open
        if circuit_breakers[current_provider].is_open():
            logger.warning(f"Circuit breaker for {current_provider} is open, skipping")
            continue
        
        # Select the appropriate function based on provider
        llm_func = call_openai if current_provider == "openai" else call_azure
        
        try:
            # Increment request counter
            LLM_REQUESTS.labels(provider=current_provider).inc()
            
            # Make the API call with retry logic
            response = await with_retry(llm_func, prompt, **kwargs)
            
            # Record the success in the circuit breaker
            circuit_breakers[current_provider].record_success()
            
            # Cache the successful response
            await cache_response(prompt, response)
            
            return response
            
        except Exception as e:
            last_error = e
            logger.error(f"Error with {current_provider}: {str(e)}")
            
            # Record the failure in the circuit breaker
            circuit_breakers[current_provider].record_failure()
    
    # If we've tried all providers and none worked
    raise AllProvidersDownException("All LLM providers are currently unavailable") from last_error

# HTTP server for Prometheus metrics
async def start_metrics_server(port: int = 8000) -> None:
    """
    Start the Prometheus metrics HTTP server.
    
    Args:
        port: The port to expose metrics on (default: 8000)
    """
    start_http_server(port)
    logger.info(f"Started Prometheus metrics server on port {port}")

# Demo function to showcase the router
async def demo() -> None:
    """Run a demonstration of the LLM router functionality."""
    # Start the metrics server
    await start_metrics_server()
    
    # Test a successful request
    try:
        sample_prompt = "Explain the concept of a circuit breaker pattern in software architecture."
        print(f"\nSending test prompt: '{sample_prompt}'")
        
        response = await route_request(sample_prompt)
        print(f"\nResponse received:\n{response}\n")
        
        # Show the cached response
        print("Testing cache by sending the same prompt again...")
        cached = await route_request(sample_prompt)
        print(f"Cache working: {'Yes' if cached == response else 'No'}")
        
    except Exception as e:
        print(f"Error during test request: {str(e)}")
    
    print("\nMetrics server running on http://localhost:8000/metrics")
    print("Press Ctrl+C to exit...")

if __name__ == "__main__":
    asyncio.run(demo())
