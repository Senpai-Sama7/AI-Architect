#!/usr/bin/env python3
"""
Secure Shell Tools Module

This module provides a secure sandbox for executing shell commands
and Python code in an isolated environment using Docker containers.

Key features:
- Isolated execution environment for untrusted code
- Resource limitations and monitoring
- Controlled access to files and network
- Proper handling of input/output
"""

import subprocess
import logging
import tempfile
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

import docker
from docker.errors import DockerException, ContainerError, ImageNotFound

from ..config.config_manager import get_config

logger = logging.getLogger(__name__)

class SecureShellTool:
    """
    Tool for safely executing shell commands in an isolated container.
    
    This class handles creating and managing Docker containers for
    safely executing potentially untrusted code.
    """
    
    def __init__(
        self,
        container_name: Optional[str] = None,
        image_name: str = "python:3.12-slim",
        socket_path: Optional[str] = None,
        working_dir: Optional[Union[str, Path]] = None,
        resource_limits: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the secure shell tool.
        
        Args:
            container_name: Name for the container (default from config)
            image_name: Docker image to use (default: python:3.12-slim)
            socket_path: Path to Docker socket (default from config)
            working_dir: Working directory for command execution
            resource_limits: Resource limits for the container
        """
        config = get_config()
        
        self.container_name = container_name or config.infra.sandbox_container_name
        self.image_name = image_name
        self.socket_path = socket_path or config.infra.docker_socket_path
        self.working_dir = working_dir or Path.cwd()
        
        # Default resource limits
        self.resource_limits = {
            "memory": "1g",
            "cpu_shares": 1024,  # CPU shares (1024 = 1 CPU)
            "pids_limit": 100,   # Maximum number of processes
            "network": "bridge"  # Network mode (bridge allows outbound)
        }
        
        # Update with user-provided limits if any
        if resource_limits:
            self.resource_limits.update(resource_limits)
        
        # Connect to Docker
        try:
            self.client = docker.from_env()
            logger.info("Connected to Docker")
        except DockerException as e:
            logger.error(f"Failed to connect to Docker: {str(e)}")
            raise
        
        # Track if the container is running
        self.container = None
        self._ensure_container()
    
    def _ensure_container(self) -> None:
        """Ensure the sandbox container exists and is ready."""
        # Check if container already exists
        try:
            self.container = self.client.containers.get(self.container_name)
            
            # Check if it's running
            if self.container.status != "running":
                logger.info(f"Starting existing container: {self.container_name}")
                self.container.start()
        
        except docker.errors.NotFound:
            # Container doesn't exist, need to create it
            logger.info(f"Creating new sandbox container: {self.container_name}")
            
            # Check if image exists or pull it
            try:
                self.client.images.get(self.image_name)
            except ImageNotFound:
                logger.info(f"Pulling Docker image: {self.image_name}")
                self.client.images.pull(self.image_name)
            
            # Create and start the container
            self.container = self.client.containers.run(
                self.image_name,
                name=self.container_name,
                command="tail -f /dev/null",  # Keep container running
                detach=True,
                volumes={
                    str(self.working_dir): {
                        "bind": "/workspace",
                        "mode": "rw"
                    }
                },
                working_dir="/workspace",
                mem_limit=self.resource_limits["memory"],
                cpu_shares=self.resource_limits["cpu_shares"],
                pids_limit=self.resource_limits["pids_limit"],
                network_mode=self.resource_limits["network"],
                cap_drop=["ALL"],  # Drop all capabilities for security
                security_opt=["no-new-privileges"],
                restart_policy={"Name": "unless-stopped"}
            )
            
            # Install common dependencies that might be needed
            self._install_basic_dependencies()
    
    def _install_basic_dependencies(self) -> None:
        """Install basic dependencies in the container."""
        logger.info("Installing basic dependencies in sandbox container")
        
        # Basic update and utility installation
        update_cmd = "apt-get update && apt-get install -y --no-install-recommends curl ca-certificates"
        
        try:
            result = self.container.exec_run(update_cmd)
            if result.exit_code != 0:
                logger.warning(f"Failed to install basic dependencies: {result.output.decode()}")
        except Exception as e:
            logger.error(f"Error installing dependencies: {str(e)}")
    
    def run(self, command: str, timeout: int = 60, capture_output: bool = True) -> Dict[str, Any]:
        """
        Run a shell command in the secure sandbox.
        
        Args:
            command: The shell command to execute
            timeout: Maximum execution time in seconds
            capture_output: Whether to capture and return command output
            
        Returns:
            Dictionary with command execution results
        """
        # Ensure container is available
        self._ensure_container()
        
        # Log the command (but sanitize any secrets)
        safe_command = command
        if "password" in safe_command.lower() or "key" in safe_command.lower():
            safe_command = "<SENSITIVE COMMAND REDACTED>"
        logger.info(f"Executing in sandbox: {safe_command}")
        
        start_time = time.time()
        
        try:
            # Execute the command
            exec_result = self.container.exec_run(
                command,
                workdir="/workspace",
                demux=True,  # Separate stdout and stderr
            )
            
            # Process results
            stdout = exec_result.output[0].decode("utf-8") if exec_result.output[0] else ""
            stderr = exec_result.output[1].decode("utf-8") if exec_result.output[1] else ""
            exit_code = exec_result.exit_code
            duration = time.time() - start_time
            
            result = {
                "success": exit_code == 0,
                "exit_code": exit_code,
                "stdout": stdout if capture_output else "<output not captured>",
                "stderr": stderr if capture_output else "<errors not captured>",
                "duration_seconds": round(duration, 3)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing command in sandbox: {str(e)}")
            
            return {
                "success": False,
                "exit_code": -1,
                "stdout": "",
                "stderr": str(e),
                "duration_seconds": round(time.time() - start_time, 3),
                "error": str(e)
            }
    
    def run_python(self, code: str, timeout: int = 60) -> Dict[str, Any]:
        """
        Run Python code in the secure sandbox.
        
        Args:
            code: The Python code to execute
            timeout: Maximum execution time in seconds
            
        Returns:
            Dictionary with code execution results
        """
        # Create a temporary file to hold the code
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name
        
        try:
            # Copy the file to the container
            dest_path = f"/tmp/{os.path.basename(temp_file_path)}"
            
            # Use the Docker API to copy the file to the container
            with open(temp_file_path, "rb") as src_file:
                data = src_file.read()
                self.container.put_archive("/tmp", data)
            
            # Run the Python script
            result = self.run(f"python {dest_path}", timeout=timeout)
            
            # Clean up the temporary file in the container
            self.container.exec_run(f"rm {dest_path}")
            
            return result
            
        finally:
            # Clean up the local temporary file
            os.unlink(temp_file_path)
    
    def install_package(self, package_name: str) -> Dict[str, Any]:
        """
        Install a Python package in the sandbox.
        
        Args:
            package_name: Name of the package to install
            
        Returns:
            Dictionary with installation results
        """
        # Sanitize the package name to prevent command injection
        if not all(c.isalnum() or c in "._-=<>+" for c in package_name):
            return {
                "success": False,
                "exit_code": -1,
                "stdout": "",
                "stderr": "Invalid package name",
                "duration_seconds": 0,
                "error": "Package name contains invalid characters"
            }
        
        # Install the package
        return self.run(f"pip install --no-cache-dir {package_name}")
    
    def cleanup(self) -> None:
        """Stop and remove the sandbox container."""
        if self.container:
            logger.info(f"Cleaning up sandbox container: {self.container_name}")
            try:
                self.container.stop()
                self.container.remove()
                self.container = None
            except Exception as e:
                logger.error(f"Error during container cleanup: {str(e)}")

class FileIOTool:
    """
    Tool for safely reading from and writing to files.
    
    This class provides methods for controlled file access with
    proper security checks.
    """
    
    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the file I/O tool.
        
        Args:
            base_dir: Base directory to restrict file operations to
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
    
    def _validate_path(self, file_path: Union[str, Path]) -> Path:
        """
        Validate that a file path is within the allowed base directory.
        
        Args:
            file_path: The file path to validate
            
        Returns:
            The absolute Path object if valid
            
        Raises:
            ValueError: If the path is outside the base directory
        """
        path = Path(file_path).resolve()
        
        if not str(path).startswith(str(self.base_dir.resolve())):
            raise ValueError(f"Access denied: {path} is outside the allowed directory")
        
        return path
    
    def read_file(self, file_path: Union[str, Path], binary: bool = False) -> Union[str, bytes]:
        """
        Read the contents of a file safely.
        
        Args:
            file_path: Path to the file to read
            binary: Whether to read in binary mode
            
        Returns:
            The file contents (str or bytes depending on mode)
            
        Raises:
            ValueError: If the path is outside the base directory
            FileNotFoundError: If the file doesn't exist
            IOError: For other I/O errors
        """
        path = self._validate_path(file_path)
        
        mode = "rb" if binary else "r"
        with open(path, mode) as f:
            return f.read()
    
    def write_file(self, file_path: Union[str, Path], content: Union[str, bytes], binary: bool = False) -> None:
        """
        Write content to a file safely.
        
        Args:
            file_path: Path to the file to write
            content: Content to write to the file
            binary: Whether to write in binary mode
            
        Raises:
            ValueError: If the path is outside the base directory
            IOError: For I/O errors
        """
        path = self._validate_path(file_path)
        
        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        mode = "wb" if binary else "w"
        with open(path, mode) as f:
            f.write(content)
    
    def list_directory(self, dir_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        List the contents of a directory safely.
        
        Args:
            dir_path: Path to the directory to list
            
        Returns:
            List of dictionaries with file information
            
        Raises:
            ValueError: If the path is outside the base directory
            NotADirectoryError: If the path is not a directory
            FileNotFoundError: If the directory doesn't exist
        """
        path = self._validate_path(dir_path)
        
        if not path.is_dir():
            raise NotADirectoryError(f"{path} is not a directory")
        
        result = []
        for item in path.iterdir():
            item_info = {
                "name": item.name,
                "path": str(item),
                "is_dir": item.is_dir(),
                "size": item.stat().st_size if item.is_file() else None,
                "modified": item.stat().st_mtime
            }
            result.append(item_info)
        
        return result

def test_sandbox():
    """Test the sandbox functionality."""
    shell_tool = SecureShellTool()
    
    print("Testing secure shell execution...")
    result = shell_tool.run("echo 'Hello from sandbox' && ls -la")
    print(f"Success: {result['success']}")
    print(f"Output: {result['stdout']}")
    
    print("\nTesting Python code execution...")
    code = """
import os
import platform

print(f"Python version: {platform.python_version()}")
print(f"Current directory: {os.getcwd()}")
print("Files in current directory:")
for item in os.listdir():
    print(f"  - {item}")
"""
    result = shell_tool.run_python(code)
    print(f"Success: {result['success']}")
    print(f"Output: {result['stdout']}")
    
    print("\nCleaning up...")
    shell_tool.cleanup()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_sandbox()
