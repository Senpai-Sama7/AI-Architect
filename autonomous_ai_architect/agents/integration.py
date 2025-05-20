#!/usr/bin/env python3
"""
Agent Integration Module

This module provides functions to integrate the agent orchestration system
with the main application. It handles task routing, error recovery, and
result processing.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union

from autonomous_ai_architect.agents.main_crew import AgentOrchestrator
from autonomous_ai_architect.core.knowledge_base import KnowledgeBase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AgentIntegration:
    """Integrates agent orchestration with the main application."""
    
    def __init__(
        self,
        redis_url: str,
        knowledge_base: Optional[KnowledgeBase] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the agent integration.
        
        Args:
            redis_url: Redis connection URL
            knowledge_base: Knowledge base instance
            config: Additional configuration parameters
        """
        self.config = config or {}
        self.orchestrator = AgentOrchestrator(
            redis_url=redis_url,
            knowledge_base=knowledge_base
        )
        logger.info("Agent integration initialized")
        
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single task with error handling and recovery.
        
        Args:
            task: Task definition including type and parameters
            
        Returns:
            Task execution results
        """
        max_retries = self.config.get("max_retries", 3)
        retry_delay = self.config.get("retry_delay", 5)  # seconds
        
        for attempt in range(max_retries):
            try:
                result = await self.orchestrator.process_task(task)
                
                if result.get("status") == "success":
                    return result
                else:
                    logger.warning(
                        f"Task execution failed (attempt {attempt+1}/{max_retries}): {result.get('error')}"
                    )
                    
                    # Add retry information to task
                    task["retry_count"] = attempt + 1
                    
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                    else:
                        return {
                            "error": f"Failed after {max_retries} attempts: {result.get('error')}",
                            "status": "failure",
                            "last_result": result
                        }
            except Exception as e:
                logger.error(f"Exception during task execution (attempt {attempt+1}/{max_retries}): {str(e)}")
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    return {
                        "error": f"Exception after {max_retries} attempts: {str(e)}",
                        "status": "failure"
                    }
                    
    async def execute_workflow(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute a sequence of tasks as a workflow.
        
        Args:
            tasks: List of task definitions
            
        Returns:
            List of task execution results
        """
        results = []
        
        for task in tasks:
            task_result = await self.execute_task(task)
            results.append(task_result)
            
            # Stop workflow execution on failure if configured
            if (
                task_result.get("status") == "failure" 
                and self.config.get("stop_on_failure", True)
            ):
                logger.warning(f"Workflow execution stopped due to task failure: {task_result.get('error')}")
                break
                
        return results
                
    async def execute_crew(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute tasks using a CrewAI crew for collaborative execution.
        
        Args:
            tasks: List of task definitions
            
        Returns:
            Crew execution results
        """
        try:
            return await self.orchestrator.create_crew(tasks)
        except Exception as e:
            logger.error(f"Crew execution failed: {str(e)}")
            return {
                "error": str(e),
                "status": "failure"
            }
            
    async def close(self):
        """Clean up resources."""
        await self.orchestrator.close()
        
# Example usage
async def main():
    from autonomous_ai_architect.core.knowledge_base import KnowledgeBase
    from autonomous_ai_architect.tools.embedding_generator import EmbeddingGenerator
    from autonomous_ai_architect.tools.vector_store import VectorStore
    
    # Create components
    vector_store = VectorStore(host="localhost", port=6333)
    embedding_generator = EmbeddingGenerator(model="text-embedding-3-small")
    
    # Initialize knowledge base
    knowledge_base = KnowledgeBase(
        vector_store=vector_store,
        embedding_generator=embedding_generator,
        collection_name="ai_architect_knowledge"
    )
    
    # Initialize integration
    integration = AgentIntegration(
        redis_url="redis://localhost:6379/0",
        knowledge_base=knowledge_base,
        config={
            "max_retries": 3,
            "retry_delay": 5,
            "stop_on_failure": True
        }
    )
    
    # Example workflow
    workflow = [
        {
            "type": "hardware_analysis",
            "prompt": "Analyze the current hardware capabilities",
            "id": "task-001"
        },
        {
            "type": "architecture",
            "prompt": "Design an AI system architecture optimized for the analyzed hardware",
            "id": "task-002"
        },
        {
            "type": "knowledge",
            "operation": "store",
            "content": "Example knowledge content",
            "metadata": {"category": "example"},
            "id": "task-003"
        }
    ]
    
    # Execute workflow
    results = await integration.execute_workflow(workflow)
    print(json.dumps(results, indent=2))
    
    # Clean up
    await integration.close()

if __name__ == "__main__":
    asyncio.run(main())
