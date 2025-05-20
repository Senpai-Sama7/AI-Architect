#!/usr/bin/env python3
"""
Agent Task Definitions and Utilities

This module provides definitions and utilities for creating and validating
agent tasks. It also includes pre-defined task templates for common operations.
"""

import json
import uuid
from typing import Dict, Any, Optional, List, Union

from pydantic import BaseModel, Field, validator

class TaskBase(BaseModel):
    """Base model for all agent tasks."""
    id: str = Field(default_factory=lambda: f"task-{uuid.uuid4().hex[:8]}")
    type: str
    
    @validator("type")
    def validate_task_type(cls, v):
        """Validate the task type."""
        valid_types = [
            "architecture", 
            "hardware_analysis", 
            "code_execution", 
            "knowledge"
        ]
        if v not in valid_types:
            raise ValueError(f"Invalid task type: {v}. Must be one of {valid_types}")
        return v

class ArchitectureTask(TaskBase):
    """Task for architecture design."""
    type: str = "architecture"
    prompt: str
    requirements: Optional[str] = None
    constraints: Optional[str] = None

class HardwareAnalysisTask(TaskBase):
    """Task for hardware analysis."""
    type: str = "hardware_analysis"
    prompt: str

class CodeExecutionTask(TaskBase):
    """Task for code execution."""
    type: str = "code_execution"
    code: str
    language: str = "python"
    timeout: int = 60

class KnowledgeTask(TaskBase):
    """Task for knowledge base operations."""
    type: str = "knowledge"
    operation: str
    query: Optional[str] = None
    content: Optional[str] = None
    document_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    limit: int = 5
    
    @validator("operation")
    def validate_operation(cls, v):
        """Validate the knowledge base operation."""
        valid_operations = ["store", "search", "retrieve"]
        if v not in valid_operations:
            raise ValueError(f"Invalid operation: {v}. Must be one of {valid_operations}")
        return v

# Task templates

def create_architecture_task(
    prompt: str,
    requirements: Optional[str] = None,
    constraints: Optional[str] = None,
    task_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create an architecture design task.
    
    Args:
        prompt: Main task prompt
        requirements: Specific requirements
        constraints: Specific constraints
        task_id: Optional task ID
        
    Returns:
        Dictionary with task definition
    """
    task = ArchitectureTask(
        prompt=prompt,
        requirements=requirements,
        constraints=constraints
    )
    
    if task_id:
        task.id = task_id
        
    return task.dict()

def create_hardware_analysis_task(
    prompt: str,
    task_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a hardware analysis task.
    
    Args:
        prompt: Analysis prompt
        task_id: Optional task ID
        
    Returns:
        Dictionary with task definition
    """
    task = HardwareAnalysisTask(prompt=prompt)
    
    if task_id:
        task.id = task_id
        
    return task.dict()

def create_code_execution_task(
    code: str,
    language: str = "python",
    timeout: int = 60,
    task_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a code execution task.
    
    Args:
        code: Code to execute
        language: Programming language
        timeout: Execution timeout in seconds
        task_id: Optional task ID
        
    Returns:
        Dictionary with task definition
    """
    task = CodeExecutionTask(
        code=code,
        language=language,
        timeout=timeout
    )
    
    if task_id:
        task.id = task_id
        
    return task.dict()

def create_knowledge_search_task(
    query: str,
    limit: int = 5,
    task_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a knowledge base search task.
    
    Args:
        query: Search query
        limit: Maximum number of results
        task_id: Optional task ID
        
    Returns:
        Dictionary with task definition
    """
    task = KnowledgeTask(
        operation="search",
        query=query,
        limit=limit
    )
    
    if task_id:
        task.id = task_id
        
    return task.dict()

def create_knowledge_store_task(
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    task_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a knowledge base storage task.
    
    Args:
        content: Content to store
        metadata: Additional metadata
        task_id: Optional task ID
        
    Returns:
        Dictionary with task definition
    """
    task = KnowledgeTask(
        operation="store",
        content=content,
        metadata=metadata or {}
    )
    
    if task_id:
        task.id = task_id
        
    return task.dict()

def create_knowledge_retrieve_task(
    document_id: str,
    task_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a knowledge base retrieval task.
    
    Args:
        document_id: ID of the document to retrieve
        task_id: Optional task ID
        
    Returns:
        Dictionary with task definition
    """
    task = KnowledgeTask(
        operation="retrieve",
        document_id=document_id
    )
    
    if task_id:
        task.id = task_id
        
    return task.dict()

def create_workflow(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create a workflow from a list of tasks.
    
    Args:
        tasks: List of task definitions
        
    Returns:
        Validated list of tasks
    """
    validated_tasks = []
    
    for task in tasks:
        task_type = task.get("type")
        
        if task_type == "architecture":
            validated_task = ArchitectureTask(**task).dict()
        elif task_type == "hardware_analysis":
            validated_task = HardwareAnalysisTask(**task).dict()
        elif task_type == "code_execution":
            validated_task = CodeExecutionTask(**task).dict()
        elif task_type == "knowledge":
            validated_task = KnowledgeTask(**task).dict()
        else:
            raise ValueError(f"Invalid task type: {task_type}")
            
        validated_tasks.append(validated_task)
        
    return validated_tasks

def save_workflow(workflow: List[Dict[str, Any]], file_path: str) -> None:
    """
    Save a workflow to a JSON file.
    
    Args:
        workflow: List of task definitions
        file_path: Path to save the workflow
    """
    with open(file_path, 'w') as f:
        json.dump(workflow, f, indent=2)
        
def load_workflow(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a workflow from a JSON file.
    
    Args:
        file_path: Path to the workflow file
        
    Returns:
        List of task definitions
    """
    with open(file_path, 'r') as f:
        workflow = json.load(f)
        
    return create_workflow(workflow)

# Example usage
if __name__ == "__main__":
    # Create a workflow
    workflow = [
        create_hardware_analysis_task(
            prompt="Analyze the current hardware capabilities for AI workloads"
        ),
        create_architecture_task(
            prompt="Design an AI system optimized for the analyzed hardware",
            requirements="Must support large language models and vector databases",
            constraints="Limited to 16GB RAM and 4 CPU cores"
        ),
        create_code_execution_task(
            code="import platform; print(platform.processor())"
        ),
        create_knowledge_store_task(
            content="Example knowledge content",
            metadata={"category": "example"}
        )
    ]
    
    # Save the workflow
    save_workflow(workflow, "example_workflow.json")
    
    # Load the workflow
    loaded_workflow = load_workflow("example_workflow.json")
    
    # Print the workflow
    print(json.dumps(loaded_workflow, indent=2))
