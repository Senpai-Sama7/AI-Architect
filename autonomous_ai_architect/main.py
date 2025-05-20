#!/usr/bin/env python3
"""
Main entry point for the Autonomous AI Architect system.

This module initializes the core system components, sets up logging,
and provides the main execution loop.
"""

import asyncio
import argparse
import logging
import signal
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

from .config.config_manager import Config, ConfigManager
from .core.llm_router import start_metrics_server, route_request
from .tools.hardware_audit import generate_hardware_report
from .core.knowledge_base import KnowledgeBase
from .tools.vector_store import VectorStore
from .tools.embedding_generator import EmbeddingGenerator
from .agents.integration import AgentIntegration

logger = logging.getLogger(__name__)

class AutonomousAIArchitect:
    """Main system controller for the Autonomous AI Architect."""
    
    def __init__(self, config: Config):
        """
        Initialize the Autonomous AI Architect system.
        
        Args:
            config: System configuration object
        """
        self.config = config
        self.shutdown_requested = False
        self.hardware_report = None
        self.knowledge_base = None
        self.agent_integration = None
        
        # Initialize subsystems based on configuration
        self._setup_logging()
        
        logger.info("Initializing Autonomous AI Architect system")
        logger.info(f"Environment: {config.environment}")
        logger.info(f"Debug mode: {'enabled' if config.debug else 'disabled'}")
    
    def _setup_logging(self) -> None:
        """Configure logging based on configuration."""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("autonomous_ai_architect.log")
            ]
        )
        
        # Set log levels for specific modules
        if self.config.debug:
            logging.getLogger("autonomous_ai_architect").setLevel(logging.DEBUG)
        else:
            # Reduce verbosity of some libraries in non-debug mode
            logging.getLogger("urllib3").setLevel(logging.WARNING)
            logging.getLogger("docker").setLevel(logging.WARNING)
    
    async def run_hardware_audit(self) -> Dict[str, Any]:
        """
        Run a hardware audit to determine system capabilities.
        
        Returns:
            Dictionary with hardware audit results
        """
        logger.info("Running hardware capability audit...")
        
        # Generate the hardware report
        report = generate_hardware_report()
        
        # Log a summary of capabilities
        logger.info("Hardware capability summary:")
        for cap_name, cap_info in report["capabilities"].items():
            status = "SUPPORTED" if cap_info["supported"] else "NOT SUPPORTED"
            logger.info(f"  {cap_name}: {status}")
        
        return report
    
    async def initialize_subsystems(self) -> None:
        """Initialize all required subsystems."""
        # Start Prometheus metrics server
        logger.info(f"Starting metrics server on port {self.config.infra.metrics_port}")
        await start_metrics_server(self.config.infra.metrics_port)
        
        # Initialize vector store
        logger.info("Initializing vector store...")
        vector_store = VectorStore(
            host=self.config.qdrant.host,
            port=self.config.qdrant.port
        )
        
        # Initialize embedding generator
        logger.info("Initializing embedding generator...")
        embedding_generator = EmbeddingGenerator(
            model=self.config.llm.embedding_model,
            api_key=self.config.llm.openai_api_key
        )
        
        # Initialize knowledge base
        logger.info("Initializing knowledge base...")
        self.knowledge_base = KnowledgeBase(
            vector_store=vector_store,
            embedding_generator=embedding_generator,
            collection_name=self.config.knowledge_base.collection_name
        )
        
        # Initialize agent integration
        logger.info("Initializing agent orchestration...")
        self.agent_integration = AgentIntegration(
            redis_url=self.config.infra.redis_url,
            knowledge_base=self.knowledge_base,
            config={
                "max_retries": self.config.agents.max_retries,
                "retry_delay": self.config.agents.retry_delay,
                "stop_on_failure": self.config.agents.stop_on_failure
            }
        )s_url,
            knowledge_base=self.knowledge_base,
            config={
                "max_retries": self.config.agents.max_retries,
                "retry_delay": self.config.agents.retry_delay,
                "stop_on_failure": self.config.agents.stop_on_failure
            }
        )
        
        logger.info("All subsystems initialized successfully")
    
    def register_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown."""
        def handle_shutdown(signum, frame):
            """Signal handler for shutdown signals."""
            sig_name = signal.Signals(signum).name
            logger.info(f"Received {sig_name} signal. Initiating graceful shutdown...")
            self.shutdown_requested = True
        
        # Register for SIGINT (Ctrl+C) and SIGTERM
        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)
    
    async def run(self) -> None:
        """Run the main system loop."""
        try:
            # Register signal handlers
            self.register_signal_handlers()
            
            # Run hardware audit
            self.hardware_report = await self.run_hardware_audit()
            
            # Initialize subsystems
            await self.initialize_subsystems()
            
            # Store hardware report in knowledge base
            if self.knowledge_base:
                logger.info("Storing hardware report in knowledge base...")
                await self.knowledge_base.store_document(
                    content=json.dumps(self.hardware_report, indent=2),
                    metadata={
                        "type": "hardware_report",
                        "timestamp": asyncio.get_event_loop().time(),
                        "source": "system_initialization"
                    }
                )
            
            # Run initial system assessment using agents
            if self.agent_integration:
                logger.info("Running initial system assessment...")
                assessment_task = {
                    "type": "hardware_analysis",
                    "prompt": "Analyze the current hardware capabilities and make recommendations for AI workloads",
                    "id": "initial-assessment"
                }
                
                assessment_result = await self.agent_integration.execute_task(assessment_task)
                logger.info(f"Initial assessment completed with status: {assessment_result.get('status')}")
            
            # Main system loop
            logger.info("Autonomous AI Architect system is ready")
            
            # Run until shutdown is requested
            while not self.shutdown_requested:
                # Process any pending tasks
                await self.process_pending_tasks()
                
                # Wait for a short interval
                await asyncio.sleep(1)
            
            # Graceful shutdown
            logger.info("Shutting down...")
            
        except Exception as e:
            logger.error(f"Unhandled exception in main loop: {str(e)}", exc_info=True)
            sys.exit(1)
            
    async def process_pending_tasks(self) -> None:
        """Process any pending tasks in the task queue."""
        # This would normally check a queue (Redis, database, etc.) for pending tasks
        # For now, this is a placeholder
        pass
    
    async def shutdown(self) -> None:
        """Perform graceful shutdown of all subsystems."""
        logger.info("Performing graceful shutdown...")
        
        # Clean up agent integration
        if self.agent_integration:
            logger.info("Shutting down agent integration...")
            await self.agent_integration.close()
        
        # Clean up other resources
        # ...
        
        logger.info("Shutdown complete")

async def main_async(args: argparse.Namespace) -> None:
    """
    Asynchronous main entry point.
    
    Args:
        args: Command line arguments
    """
    # Load configuration
    config_manager = ConfigManager(config_path=args.config)
    config = config_manager.config
    
    # Override config with command line arguments
    if args.debug:
        config_manager.set_value('debug', True)
    if args.log_level:
        config_manager.set_value('log_level', args.log_level)
    
    # Initialize the system
    system = AutonomousAIArchitect(config)
    
    try:
        # Run the system
        await system.run()
    finally:
        # Ensure cleanup happens
        await system.shutdown()

def main() -> None:
    """
    Synchronous main entry point for command-line use.
    
    Parses command line arguments and runs the system.
    """
    parser = argparse.ArgumentParser(description="Autonomous AI Architect System")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                      help="Set logging level")
    parser.add_argument("--hardware-audit", action="store_true", 
                      help="Run a hardware audit and exit")
    
    args = parser.parse_args()
    
    # If we're just running a hardware audit
    if args.hardware_audit:
        report = asyncio.run(generate_hardware_report_async())
        print(json.dumps(report, indent=2))
        return
    
    # Run the main system
    asyncio.run(main_async(args))

async def generate_hardware_report_async() -> Dict[str, Any]:
    """
    Run a hardware audit asynchronously.
    
    Returns:
        Dictionary with hardware audit results
    """
    return generate_hardware_report()

if __name__ == "__main__":
    main()
