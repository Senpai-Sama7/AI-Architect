#!/usr/bin/env python3
"""
Command-Line Interface for Autonomous AI Architect Agent System

This module provides a command-line interface for interacting directly
with the agent system. It allows users to submit tasks, view task status,
and get results from the agent system.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

from autonomous_ai_architect.config.config_manager import ConfigManager
from autonomous_ai_architect.core.knowledge_base import KnowledgeBase
from autonomous_ai_architect.tools.embedding_generator import EmbeddingGenerator
from autonomous_ai_architect.tools.vector_store import VectorStore
from autonomous_ai_architect.agents.integration import AgentIntegration
from autonomous_ai_architect.utils.cli_output import (
    print_header, print_success, print_error, print_warning,
    print_info, print_progress, print_task, format_json, clear_screen
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class AgentCLI:
    """Command-line interface for interacting with agents."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the agent CLI.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config_manager = ConfigManager(config_path=config_path)
        self.config = self.config_manager.config
        
        # Initialize components
        self.knowledge_base = None
        self.agent_integration = None
        
        logger.info("Agent CLI initialized")
        
    async def initialize(self):
        """Initialize agent system components."""
        logger.info("Initializing agent system components...")
        
        # Initialize vector store
        vector_store = VectorStore(
            host=self.config.qdrant.host,
            port=self.config.qdrant.port
        )
        
        # Initialize embedding generator
        embedding_generator = EmbeddingGenerator(
            model=self.config.llm.embedding_model,
            api_key=self.config.llm.openai_api_key
        )
        
        # Initialize knowledge base
        self.knowledge_base = KnowledgeBase(
            vector_store=vector_store,
            embedding_generator=embedding_generator,
            collection_name=self.config.knowledge_base.collection_name
        )
        
        # Initialize agent integration
        self.agent_integration = AgentIntegration(
            redis_url=self.config.infra.redis_url,
            knowledge_base=self.knowledge_base,
            config={
                "max_retries": self.config.agents.max_retries,
                "retry_delay": self.config.agents.retry_delay,
                "stop_on_failure": self.config.agents.stop_on_failure
            }
        )
        
        logger.info("Agent system components initialized")
        
    async def close(self):
        """Clean up resources."""
        if self.agent_integration:
            await self.agent_integration.close()
    
    async def run_task(self, task_type: str, prompt: str, task_id: Optional[str] = None, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a single agent task.
        
        Args:
            task_type: Type of task to run
            prompt: Task prompt or input
            task_id: Optional task ID
            output_file: Optional file to save results
            
        Returns:
            Task execution results
        """
        # Initialize components if not already done
        if not self.agent_integration:
            print_progress("Initializing agent system components...")
            await self.initialize()
            
        # Create task definition
        task = {
            "type": task_type,
            "prompt": prompt,
            "id": task_id or f"cli-task-{int(asyncio.get_event_loop().time())}"
        }
        
        logger.info(f"Running task: {task['id']} ({task_type})")
        print_header(f"Processing Task: {task_type}")
        print_info(f"Task ID: {task['id']}")
        print_progress("Executing task...")
        
        try:
            # Execute task
            result = await self.agent_integration.execute_task(task)
            
            # Display task status
            if result.get("status") == "success":
                print_success(f"Task completed successfully")
            else:
                print_error(f"Task failed: {result.get('error', 'Unknown error')}")
            
            # Save results to file if requested
            if output_file and result.get("status") == "success":
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                logger.info(f"Results saved to {output_file}")
                print_success(f"Results saved to {output_file}")
            
            return result
        except Exception as e:
            logger.error(f"Task execution failed: {str(e)}")
            print_error(f"Task execution failed: {str(e)}")
            return {
                "error": str(e),
                "status": "failure"
            }
    
    async def run_workflow(self, workflow_file: str, output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Run a workflow of multiple tasks from a file.
        
        Args:
            workflow_file: Path to workflow JSON file
            output_dir: Optional directory to save results
            
        Returns:
            List of task execution results
        """
        # Initialize components if not already done
        if not self.agent_integration:
            await self.initialize()
            
        try:
            # Load workflow from file
            with open(workflow_file, 'r') as f:
                workflow = json.load(f)
                
            if not isinstance(workflow, list):
                raise ValueError("Workflow must be a JSON array of tasks")
                
            logger.info(f"Running workflow with {len(workflow)} tasks")
            print(f"⏳ Running workflow with {len(workflow)} tasks...", file=sys.stderr)
            
            # Create output directory if needed
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
            # Execute workflow
            results = await self.agent_integration.execute_workflow(workflow)
            
            # Save results to files if requested
            if output_dir:
                for i, result in enumerate(results):
                    output_file = os.path.join(output_dir, f"task_{i+1}.json")
                    with open(output_file, 'w') as f:
                        json.dump(result, f, indent=2)
                logger.info(f"Results saved to {output_dir}")
                print(f"✅ Results saved to {output_dir}", file=sys.stderr)
            
            return results
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            return [{
                "error": str(e),
                "status": "failure"
            }]
    
    async def run_crew(self, tasks_file: str, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Run tasks using a CrewAI crew.
        
        Args:
            tasks_file: Path to tasks JSON file
            output_file: Optional file to save results
            
        Returns:
            Crew execution results
        """
        # Initialize components if not already done
        if not self.agent_integration:
            await self.initialize()
            
        try:
            # Load tasks from file
            with open(tasks_file, 'r') as f:
                tasks = json.load(f)
                
            if not isinstance(tasks, list):
                raise ValueError("Tasks must be a JSON array")
                
            logger.info(f"Running crew with {len(tasks)} tasks")
            print(f"⏳ Running crew with {len(tasks)} tasks...", file=sys.stderr)
            
            # Execute crew
            result = await self.agent_integration.execute_crew(tasks)
            
            # Save results to file if requested
            if output_file and result.get("status") == "success":
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                logger.info(f"Results saved to {output_file}")
                print(f"✅ Results saved to {output_file}", file=sys.stderr)
            
            return result
        except Exception as e:
            logger.error(f"Crew execution failed: {str(e)}")
            return {
                "error": str(e),
                "status": "failure"
            }
    
    async def search_knowledge(self, query: str, limit: int = 5, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Search the knowledge base.
        
        Args:
            query: Search query
            limit: Maximum number of results
            output_file: Optional file to save results
            
        Returns:
            Search results
        """
        # Initialize components if not already done
        if not self.knowledge_base:
            await self.initialize()
            
        try:
            logger.info(f"Searching knowledge base: {query}")
            print(f"🔍 Searching knowledge base...", file=sys.stderr)
            
            # Search knowledge base
            results = await self.knowledge_base.search(query, limit=limit)
            
            # Format results for display
            formatted_results = {
                "query": query,
                "results": results,
                "status": "success"
            }
            
            # Save results to file if requested
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(formatted_results, f, indent=2)
                logger.info(f"Results saved to {output_file}")
                print(f"✅ Results saved to {output_file}", file=sys.stderr)
            
            return formatted_results
        except Exception as e:
            logger.error(f"Knowledge base search failed: {str(e)}")
            return {
                "error": str(e),
                "status": "failure"
            }

async def main():
    """Run the agent CLI."""
    parser = argparse.ArgumentParser(description="Autonomous AI Architect Agent CLI")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Task command
    task_parser = subparsers.add_parser("task", help="Run a single agent task")
    task_parser.add_argument("--type", "-t", required=True, 
                          choices=["architecture", "hardware_analysis", "code_execution", "knowledge"],
                          help="Type of task to run")
    task_parser.add_argument("--prompt", "-p", required=True, 
                          help="Task prompt or input")
    task_parser.add_argument("--id", "-i", help="Optional task ID")
    task_parser.add_argument("--output", "-o", help="Output file for results")
    
    # Workflow command
    workflow_parser = subparsers.add_parser("workflow", help="Run a workflow of multiple tasks")
    workflow_parser.add_argument("--file", "-f", required=True, 
                              help="Path to workflow JSON file")
    workflow_parser.add_argument("--output-dir", "-o", 
                              help="Output directory for results")
    
    # Crew command
    crew_parser = subparsers.add_parser("crew", help="Run tasks using a CrewAI crew")
    crew_parser.add_argument("--file", "-f", required=True, 
                          help="Path to tasks JSON file")
    crew_parser.add_argument("--output", "-o", 
                          help="Output file for results")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search the knowledge base")
    search_parser.add_argument("--query", "-q", required=True, 
                            help="Search query")
    search_parser.add_argument("--limit", "-l", type=int, default=5,
                            help="Maximum number of results")
    search_parser.add_argument("--output", "-o", 
                            help="Output file for results")
    
    args = parser.parse_args()
    
    # Check for no-color flag
    if args.no_color:
        # Disable colors by setting all color codes to empty strings
        from autonomous_ai_architect.utils.cli_output import Colors
        for attr in dir(Colors):
            if not attr.startswith('__'):
                setattr(Colors, attr, '')
    
    # Create agent CLI
    cli = AgentCLI(config_path=args.config)
    
    # Print header
    clear_screen()
    print_header("Autonomous AI Architect CLI")
    
    try:
        # Run the requested command
        if args.command == "task":
            print_info(f"Running task of type: {args.type}")
            result = await cli.run_task(
                task_type=args.type,
                prompt=args.prompt,
                task_id=args.id,
                output_file=args.output
            )
            print_header("Task Result")
            print(format_json(result))
            
        elif args.command == "workflow":
            print_info(f"Running workflow from file: {args.file}")
            results = await cli.run_workflow(
                workflow_file=args.file,
                output_dir=args.output_dir
            )
            print_header("Workflow Results")
            print(format_json(results))
            
        elif args.command == "crew":
            print_info(f"Running crew from file: {args.file}")
            result = await cli.run_crew(
                tasks_file=args.file,
                output_file=args.output
            )
            print_header("Crew Results")
            print(format_json(result))
            
        elif args.command == "search":
            print_info(f"Searching knowledge base for: {args.query}")
            result = await cli.search_knowledge(
                query=args.query,
                limit=args.limit,
                output_file=args.output
            )
            print_header("Search Results")
            print(format_json(result))
            
        else:
            parser.print_help()
            
    except Exception as e:
        print_error(f"Error: {str(e)}")
        logger.error(f"CLI error: {str(e)}", exc_info=True)
        sys.exit(1)
            
    finally:
        # Clean up resources
        if hasattr(cli, 'close'):
            print_info("Cleaning up resources...")
            await cli.close()
            print_success("Done")

if __name__ == "__main__":
    asyncio.run(main())
    
def main_wrapper():
    """Wrapper function for Poetry script entry point."""
    asyncio.run(main())
