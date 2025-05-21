#!/usr/bin/env python3
"""
Hardware Audit Tool for CrewAI

This module provides a CrewAI-compatible tool interface for the hardware audit
functionality, allowing agents to access hardware information.
"""

import logging
import json
from typing import Dict, Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from autonomous_ai_architect.tools.hardware_audit import HardwareAudit

logger = logging.getLogger(__name__)

class HardwareAuditInput(BaseModel):
    """Input schema for the hardware audit tool."""
    full_audit: bool = Field(
        default=True,
        description="Whether to run a full hardware audit"
    )
    specific_component: str = Field(
        default="",
        description="Specific component to audit (cpu, memory, gpu, disk, network)"
    )

class HardwareAuditTool(BaseTool):
    """Tool for auditing hardware capabilities."""
    
    name: str = "HardwareAuditTool"
    description: str = "Analyzes the hardware capabilities of the system, including CPU, memory, GPU, disk, and network"
    args_schema: type[BaseModel] = HardwareAuditInput
    
    def __init__(self):
        """Initialize the hardware audit tool."""
        super().__init__()
        self.hardware_audit = HardwareAudit()
    
    def _run(self, full_audit: bool = True, specific_component: str = "") -> str:
        """
        Run the hardware audit.
        
        Args:
            full_audit: Whether to run a full hardware audit
            specific_component: Specific component to audit
            
        Returns:
            JSON string containing audit results
        """
        logger.info("Starting hardware audit")
        try:
            if full_audit:
                result = self.hardware_audit.run_full_audit()
                logger.info("Full hardware audit completed successfully")
                return json.dumps(result, indent=2)
            elif specific_component:
                if specific_component == "cpu":
                    result = self.hardware_audit.get_cpu_info()
                elif specific_component == "memory":
                    result = self.hardware_audit.get_memory_info()
                elif specific_component == "gpu":
                    result = self.hardware_audit.get_gpu_info()
                elif specific_component == "disk":
                    result = self.hardware_audit.get_disk_info()
                elif specific_component == "network":
                    result = self.hardware_audit.get_network_info()
                else:
                    logger.warning(f"Unknown component: {specific_component}")
                    return f"Unknown component: {specific_component}"
                    
                logger.info(f"Audit for {specific_component} completed successfully")
                return json.dumps(result, indent=2)
            else:
                logger.warning("No audit type specified")
                return "Please specify either full_audit=True or a specific_component"
        except Exception as e:
            logger.error(f"Hardware audit failed: {str(e)}")
            return f"Hardware audit failed: {str(e)}"
    
    async def get_hardware_info(self) -> Dict[str, Any]:
        """
        Get hardware information asynchronously.
        
        Returns:
            Dictionary with hardware information
        """
        logger.info("Starting asynchronous hardware audit")
        try:
            # Run the audit synchronously since hardware operations are typically I/O bound
            result = self.hardware_audit.run_full_audit()
            logger.info("Asynchronous hardware audit completed successfully")
            return result
        except Exception as e:
            logger.error(f"Failed to get hardware information: {str(e)}")
            return {"error": str(e)}
