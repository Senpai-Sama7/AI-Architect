#!/usr/bin/env python3
"""
CrewAI Agent Orchestration System for Autonomous AI Architect.

This module implements a robust orchestration layer for AI agents using the CrewAI
framework. It defines specialized agents for different tasks including hardware analysis,
code execution, knowledge management, and system administration.

Each agent leverages the LLM router to dynamically select between available LLM
providers with circuit-breaking capabilities and metrics collection.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable

import redis.asyncio as redis
from crewai import Agent, Crew, Task, Process
from prometheus_client import Counter, Histogram

from autonomous_ai_architect.core.llm_router import route_request
from autonomous_ai_architect.tools.secure_shell import SecureShellTool
from autonomous_ai_architect.tools.hardware_audit_tool import HardwareAuditTool
from autonomous_ai_architect.core.knowledge_base import KnowledgeBase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
AGENT_OPERATIONS = Counter(
    "agent_operations_total",
    "Total number of agent operations",
    ["agent_type", "status"]
)
AGENT_LATENCY = Histogram(
    "agent_latency_seconds",
    "Agent operation latency in seconds",
    ["agent_type"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600]
)

class BaseAgent(Agent):
    """Base agent class with common functionality."""
    
    def __init__(self, 
                 name: str, 
                 role: str, 
                 redis_client: redis.Redis,
                 knowledge_base: Optional[KnowledgeBase] = None,
                 shell_tool: Optional[SecureShellTool] = None,
                 hardware_tool: Optional[HardwareAuditTool] = None,
                 goal: str = "",
                 backstory: str = "",
                 verbose: bool = False,
                 allow_delegation: bool = False):
        """
        Initialize base agent with common dependencies.
        
        Args:
            name: Agent name
            role: Agent role description
            redis_client: Redis client for caching and state
            knowledge_base: Knowledge base for storing and retrieving information
            shell_tool: Optional secure shell tool for sandbox operations
            hardware_tool: Optional hardware audit tool
            goal: Agent's primary objective
            backstory: Agent's background information
            verbose: Whether to log detailed information
            allow_delegation: Whether the agent can delegate tasks
        """
        self.redis_client = redis_client
        self.knowledge_base = knowledge_base
        self.shell_tool = shell_tool
        self.hardware_tool = hardware_tool
        
        tools = []
        if shell_tool:
            tools.append(shell_tool)
            
        if hardware_tool:
            tools.append(hardware_tool)
            
        super().__init__(
            name=name,
            role=role,
            goal=goal,
            backstory=backstory,
            verbose=verbose,
            tools=tools,
            allow_delegation=allow_delegation
        )

    async def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Base implementation of task execution.
        
        Args:
            task: Task dictionary with 'prompt' and other parameters
            
        Returns:
            Dictionary with processed results
        """
        raise NotImplementedError("Subclasses must implement run()")
        
    async def _llm_request(self, prompt: str, **kwargs) -> str:
        """
        Make a request to the LLM through the router.
        
        Args:
            prompt: The input prompt for the LLM
            **kwargs: Additional parameters for the LLM
            
        Returns:
            LLM response as string
        """
        return await route_request(prompt, **kwargs)

class ArchitectAgent(BaseAgent):
    """Agent responsible for high-level system design and architecture decisions."""
    
    async def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze requirements and create system architecture.
        
        Args:
            task: Task details including requirements and constraints
            
        Returns:
            Dictionary with architecture plan
        """
        prompt = f"""
        As an AI Architecture Expert, analyze the following requirements and create 
        a comprehensive system architecture plan:
        
        REQUIREMENTS:
        {task.get('requirements', 'No specific requirements provided')}
        
        CONSTRAINTS:
        {task.get('constraints', 'No specific constraints provided')}
        
        USER REQUEST:
        {task.get('prompt', 'No specific prompt provided')}
        
        Please provide:
        1. A high-level system architecture diagram (as text)
        2. Component breakdown with responsibilities
        3. Key interfaces and data flows
        4. Technology stack recommendations
        5. Security considerations
        6. Scalability and performance considerations
        7. Implementation plan with phases
        """
        
        start_time = asyncio.get_event_loop().time()
        try:
            response = await self._llm_request(prompt)
            AGENT_OPERATIONS.labels(agent_type="architect", status="success").inc()
            
            # Store architecture plan in knowledge base if available
            if self.knowledge_base:
                await self.knowledge_base.store_document(
                    content=response,
                    metadata={
                        "type": "architecture_plan",
                        "task_id": task.get("id", "unknown"),
                        "timestamp": asyncio.get_event_loop().time()
                    }
                )
            
            return {
                "architecture_plan": response,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"ArchitectAgent failed: {str(e)}")
            AGENT_OPERATIONS.labels(agent_type="architect", status="failure").inc()
            return {
                "error": str(e),
                "status": "failure"
            }
        finally:
            AGENT_LATENCY.labels(agent_type="architect").observe(
                asyncio.get_event_loop().time() - start_time
            )

class HardwareAnalysisAgent(BaseAgent):
    """Agent responsible for hardware analysis and recommendations."""
    
    async def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze hardware capabilities and make recommendations.
        
        Args:
            task: Task details
            
        Returns:
            Dictionary with hardware analysis results
        """
        if not self.hardware_tool:
            return {
                "error": "Hardware audit tool not available",
                "status": "failure"
            }
            
        start_time = asyncio.get_event_loop().time()
        try:
            # Get hardware information using the tool
            hardware_info = await self.hardware_tool.get_hardware_info()
            
            prompt = f"""
            As a Hardware Analysis Expert, please analyze the following hardware information
            and provide recommendations for AI workloads:
            
            HARDWARE INFORMATION:
            {json.dumps(hardware_info, indent=2)}
            
            USER REQUEST:
            {task.get('prompt', 'No specific prompt provided')}
            
            Please provide:
            1. Summary of current hardware capabilities
            2. CPU performance analysis
            3. Memory capacity and speed analysis
            4. Storage performance and capacity analysis
            5. GPU capabilities (if available)
            6. Network bandwidth assessment
            7. Recommendations for AI workloads
            8. Potential bottlenecks
            9. Upgrade recommendations if needed
            """
            
            analysis = await self._llm_request(prompt)
            AGENT_OPERATIONS.labels(agent_type="hardware_analysis", status="success").inc()
            
            # Store hardware analysis in knowledge base
            if self.knowledge_base:
                await self.knowledge_base.store_document(
                    content=analysis,
                    metadata={
                        "type": "hardware_analysis",
                        "task_id": task.get("id", "unknown"),
                        "timestamp": asyncio.get_event_loop().time(),
                        "raw_hardware_info": hardware_info
                    }
                )
            
            return {
                "hardware_info": hardware_info,
                "analysis": analysis,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"HardwareAnalysisAgent failed: {str(e)}")
            AGENT_OPERATIONS.labels(agent_type="hardware_analysis", status="failure").inc()
            return {
                "error": str(e),
                "status": "failure"
            }
        finally:
            AGENT_LATENCY.labels(agent_type="hardware_analysis").observe(
                asyncio.get_event_loop().time() - start_time
            )

class CodeExecutionAgent(BaseAgent):
    """Agent responsible for secure code execution in sandbox environments."""
    
    async def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute code in a secure sandbox environment.
        
        Args:
            task: Task details including code to execute
            
        Returns:
            Dictionary with execution results
        """
        if not self.shell_tool:
            return {
                "error": "Secure shell tool not available",
                "status": "failure"
            }
            
        start_time = asyncio.get_event_loop().time()
        try:
            code = task.get("code", "")
            language = task.get("language", "python")
            timeout = task.get("timeout", 60)
            
            # Analyze code for security risks
            security_prompt = f"""
            As a Security Expert, please analyze the following code for security risks:
            
            ```{language}
            {code}
            ```
            
            Please identify any:
            1. Security vulnerabilities
            2. Potential malicious operations
            3. Resource consumption issues
            4. File system access concerns
            5. Network access concerns
            
            Provide a security risk assessment (Low, Medium, High) and explain your reasoning.
            """
            
            security_analysis = await self._llm_request(security_prompt)
            
            # Execute code only if it's deemed safe
            if "High" not in security_analysis:
                result = await self.shell_tool.execute_code(
                    code=code,
                    language=language,
                    timeout=timeout
                )
                
                AGENT_OPERATIONS.labels(agent_type="code_execution", status="success").inc()
                return {
                    "result": result,
                    "security_analysis": security_analysis,
                    "status": "success"
                }
            else:
                logger.warning(f"Code execution blocked due to security concerns: {security_analysis}")
                AGENT_OPERATIONS.labels(agent_type="code_execution", status="blocked").inc()
                return {
                    "error": "Code execution blocked due to security concerns",
                    "security_analysis": security_analysis,
                    "status": "blocked"
                }
        except Exception as e:
            logger.error(f"CodeExecutionAgent failed: {str(e)}")
            AGENT_OPERATIONS.labels(agent_type="code_execution", status="failure").inc()
            return {
                "error": str(e),
                "status": "failure"
            }
        finally:
            AGENT_LATENCY.labels(agent_type="code_execution").observe(
                asyncio.get_event_loop().time() - start_time
            )

class KnowledgeAgent(BaseAgent):
    """Agent responsible for knowledge management and retrieval."""
    
    async def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store or retrieve knowledge from the knowledge base.
        
        Args:
            task: Task details including operation type and data
            
        Returns:
            Dictionary with operation results
        """
        if not self.knowledge_base:
            return {
                "error": "Knowledge base not available",
                "status": "failure"
            }
            
        start_time = asyncio.get_event_loop().time()
        try:
            operation = task.get("operation", "search")
            
            if operation == "store":
                content = task.get("content", "")
                metadata = task.get("metadata", {})
                
                document_id = await self.knowledge_base.store_document(
                    content=content,
                    metadata=metadata
                )
                
                AGENT_OPERATIONS.labels(agent_type="knowledge", status="success").inc()
                return {
                    "document_id": document_id,
                    "status": "success"
                }
                
            elif operation == "search":
                query = task.get("query", "")
                limit = task.get("limit", 5)
                
                results = await self.knowledge_base.search(
                    query=query,
                    limit=limit
                )
                
                AGENT_OPERATIONS.labels(agent_type="knowledge", status="success").inc()
                return {
                    "results": results,
                    "status": "success"
                }
                
            elif operation == "retrieve":
                document_id = task.get("document_id", "")
                
                document = await self.knowledge_base.get_document(document_id)
                
                AGENT_OPERATIONS.labels(agent_type="knowledge", status="success").inc()
                return {
                    "document": document,
                    "status": "success"
                }
                
            else:
                AGENT_OPERATIONS.labels(agent_type="knowledge", status="failure").inc()
                return {
                    "error": f"Unknown operation: {operation}",
                    "status": "failure"
                }
        except Exception as e:
            logger.error(f"KnowledgeAgent failed: {str(e)}")
            AGENT_OPERATIONS.labels(agent_type="knowledge", status="failure").inc()
            return {
                "error": str(e),
                "status": "failure"
            }
        finally:
            AGENT_LATENCY.labels(agent_type="knowledge").observe(
                asyncio.get_event_loop().time() - start_time
            )

class AgentOrchestrator:
    """Orchestrates multiple agents to perform complex tasks."""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        knowledge_base: Optional[KnowledgeBase] = None
    ):
        """
        Initialize the agent orchestrator.
        
        Args:
            redis_url: Redis connection URL
            knowledge_base: Optional knowledge base instance
        """
        self.redis_client = redis.from_url(redis_url)
        self.knowledge_base = knowledge_base
        
        # Initialize tools
        self.shell_tool = SecureShellTool()
        self.hardware_tool = HardwareAuditTool()
        
        # Initialize agents
        self.architect_agent = ArchitectAgent(
            name="Architect",
            role="System architecture expert responsible for high-level design decisions",
            redis_client=self.redis_client,
            knowledge_base=self.knowledge_base,
            goal="Create optimal system architectures for AI workloads",
            backstory="Experienced system architect with expertise in designing scalable AI systems",
            verbose=True,
            allow_delegation=True
        )
        
        self.hardware_agent = HardwareAnalysisAgent(
            name="HardwareAnalyst",
            role="Hardware expert responsible for system capability analysis",
            redis_client=self.redis_client,
            knowledge_base=self.knowledge_base,
            hardware_tool=self.hardware_tool,
            goal="Analyze hardware capabilities and provide optimization recommendations",
            backstory="Hardware specialist with deep knowledge of AI compute requirements",
            verbose=True
        )
        
        self.code_agent = CodeExecutionAgent(
            name="CodeExecutor",
            role="Secure code execution specialist",
            redis_client=self.redis_client,
            knowledge_base=self.knowledge_base,
            shell_tool=self.shell_tool,
            goal="Execute code securely in isolated environments",
            backstory="Security expert specializing in safe code execution",
            verbose=True
        )
        
        self.knowledge_agent = KnowledgeAgent(
            name="KnowledgeManager",
            role="Knowledge management specialist",
            redis_client=self.redis_client,
            knowledge_base=self.knowledge_base,
            goal="Efficiently store and retrieve knowledge",
            backstory="Information specialist with expertise in knowledge organization",
            verbose=True
        )
        
        logger.info("Agent orchestrator initialized with all agents")

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task using the appropriate agent.
        
        Args:
            task: Task details including type and parameters
            
        Returns:
            Dictionary with task results
        """
        task_type = task.get("type", "")
        
        if task_type == "architecture":
            return await self.architect_agent.run(task)
            
        elif task_type == "hardware_analysis":
            return await self.hardware_agent.run(task)
            
        elif task_type == "code_execution":
            return await self.code_agent.run(task)
            
        elif task_type == "knowledge":
            return await self.knowledge_agent.run(task)
            
        else:
            logger.error(f"Unknown task type: {task_type}")
            return {
                "error": f"Unknown task type: {task_type}",
                "status": "failure"
            }
            
    async def create_crew(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a CrewAI crew to execute a sequence of tasks.
        
        Args:
            tasks: List of task definitions
            
        Returns:
            Dictionary with results from all tasks
        """
        # Convert tasks to CrewAI Task objects
        crew_tasks = []
        
        for i, task_def in enumerate(tasks):
            agent = None
            
            # Select appropriate agent based on task type
            if task_def.get("type") == "architecture":
                agent = self.architect_agent
            elif task_def.get("type") == "hardware_analysis":
                agent = self.hardware_agent
            elif task_def.get("type") == "code_execution":
                agent = self.code_agent
            elif task_def.get("type") == "knowledge":
                agent = self.knowledge_agent
            else:
                logger.warning(f"Unknown task type: {task_def.get('type')}")
                continue
                
            # Create CrewAI Task
            crew_task = Task(
                description=task_def.get("prompt", f"Task {i}"),
                expected_output=task_def.get("expected_output", "Completed task"),
                agent=agent
            )
            
            crew_tasks.append(crew_task)
            
        # Create and run crew
        if crew_tasks:
            crew = Crew(
                agents=[
                    self.architect_agent,
                    self.hardware_agent,
                    self.code_agent,
                    self.knowledge_agent
                ],
                tasks=crew_tasks,
                verbose=True,
                process=Process.sequential  # Default to sequential execution
            )
            
            # Execute crew
            try:
                result = crew.kickoff()
                return {
                    "result": result,
                    "status": "success"
                }
            except Exception as e:
                logger.error(f"Crew execution failed: {str(e)}")
                return {
                    "error": str(e),
                    "status": "failure"
                }
        else:
            return {
                "error": "No valid tasks found",
                "status": "failure"
            }
    
    async def close(self):
        """Clean up resources."""
        await self.redis_client.close()
        
# Example usage
async def main():
    # Initialize knowledge base
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
    
    # Initialize orchestrator
    orchestrator = AgentOrchestrator(
        redis_url="redis://localhost:6379/0",
        knowledge_base=knowledge_base
    )
    
    # Example task
    task = {
        "type": "hardware_analysis",
        "prompt": "Analyze the current hardware capabilities and make recommendations for AI workloads",
        "id": "task-001"
    }
    
    # Process task
    result = await orchestrator.process_task(task)
    print(json.dumps(result, indent=2))
    
    # Clean up
    await orchestrator.close()

if __name__ == "__main__":
    asyncio.run(main())
