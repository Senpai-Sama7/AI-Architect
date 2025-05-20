#!/usr/bin/env python3
"""
Hardware Audit Module

This module provides functionality for auditing hardware capabilities
and resource availability for the Autonomous AI Architect system.

Key features:
- CPU information and capabilities
- Memory analysis
- GPU detection and capabilities
- Disk space analysis
- Network capabilities
"""

import logging
import platform
import subprocess
import json
from typing import Dict, Any, List, Optional, Tuple
import psutil
import docker

logger = logging.getLogger(__name__)

class HardwareAudit:
    """
    Hardware audit functionality for system capability assessment.
    
    This class provides methods for analyzing the hardware environment
    and determining what AI capabilities are supported.
    """
    
    def __init__(self):
        """Initialize the hardware audit module."""
        self.system_info = {}
        self.cpu_info = {}
        self.memory_info = {}
        self.gpu_info = {}
        self.disk_info = {}
        self.network_info = {}
        self.docker_info = {}
    
    def run_full_audit(self) -> Dict[str, Any]:
        """
        Run a complete hardware audit and return the results.
        
        Returns:
            Dictionary containing all audit results
        """
        self.system_info = self.get_system_info()
        self.cpu_info = self.get_cpu_info()
        self.memory_info = self.get_memory_info()
        self.gpu_info = self.get_gpu_info()
        self.disk_info = self.get_disk_info()
        self.network_info = self.get_network_info()
        self.docker_info = self.get_docker_info()
        
        # Compile all results
        results = {
            "system": self.system_info,
            "cpu": self.cpu_info,
            "memory": self.memory_info,
            "gpu": self.gpu_info,
            "disk": self.disk_info,
            "network": self.network_info,
            "docker": self.docker_info,
            "capabilities": self.determine_capabilities()
        }
        
        return results
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get general system information.
        
        Returns:
            Dictionary with system information
        """
        info = {
            "os": platform.system(),
            "os_version": platform.version(),
            "os_release": platform.release(),
            "architecture": platform.machine(),
            "hostname": platform.node(),
            "python_version": platform.python_version(),
        }
        
        # Try to get Linux distribution information if on Linux
        if info["os"] == "Linux":
            try:
                # Try to use /etc/os-release first (more modern)
                os_release = {}
                try:
                    with open("/etc/os-release") as f:
                        for line in f:
                            if "=" in line:
                                key, value = line.strip().split("=", 1)
                                os_release[key] = value.strip('"')
                    
                    if "NAME" in os_release:
                        info["distro"] = os_release["NAME"]
                    if "VERSION_ID" in os_release:
                        info["distro_version"] = os_release["VERSION_ID"]
                except FileNotFoundError:
                    # Fall back to platform.dist() for older systems
                    pass
            except Exception as e:
                logger.error(f"Error getting Linux distribution info: {str(e)}")
        
        return info
    
    def get_cpu_info(self) -> Dict[str, Any]:
        """
        Get detailed CPU information.
        
        Returns:
            Dictionary with CPU details
        """
        info = {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "current_frequency": psutil.cpu_freq().current if psutil.cpu_freq() else None,
            "min_frequency": psutil.cpu_freq().min if psutil.cpu_freq() else None,
            "max_frequency": psutil.cpu_freq().max if psutil.cpu_freq() else None,
            "cpu_percent": psutil.cpu_percent(interval=1),
            "cpu_stats": psutil.cpu_stats()._asdict(),
        }
        
        # Try to get CPU model information from /proc/cpuinfo on Linux
        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo") as f:
                    cpuinfo = f.read()
                
                # Extract model name
                for line in cpuinfo.split("\n"):
                    if "model name" in line:
                        info["model"] = line.split(":")[1].strip()
                        break
            except Exception as e:
                logger.error(f"Error reading /proc/cpuinfo: {str(e)}")
        
        # Try lscpu for even more detailed information on Linux
        if platform.system() == "Linux":
            try:
                result = subprocess.run(["lscpu"], capture_output=True, text=True, check=True)
                lscpu_info = {}
                
                for line in result.stdout.split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        lscpu_info[key.strip()] = value.strip()
                
                info["details"] = lscpu_info
                
                # Extract architecture details
                if "Architecture" in lscpu_info:
                    info["architecture"] = lscpu_info["Architecture"]
                if "CPU op-mode(s)" in lscpu_info:
                    info["op_modes"] = lscpu_info["CPU op-mode(s)"]
                if "Byte Order" in lscpu_info:
                    info["byte_order"] = lscpu_info["Byte Order"]
                
                # Extract cache information
                caches = {}
                for key, value in lscpu_info.items():
                    if "cache" in key.lower():
                        caches[key] = value
                
                if caches:
                    info["caches"] = caches
                
            except Exception as e:
                logger.error(f"Error running lscpu: {str(e)}")
        
        return info
    
    def get_memory_info(self) -> Dict[str, Any]:
        """
        Get detailed memory information.
        
        Returns:
            Dictionary with memory details
        """
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        info = {
            "total": mem.total,
            "available": mem.available,
            "used": mem.used,
            "free": mem.free,
            "percent_used": mem.percent,
            "swap_total": swap.total,
            "swap_used": swap.used,
            "swap_free": swap.free,
            "swap_percent": swap.percent,
        }
        
        # Add human-readable sizes
        info["total_gb"] = round(mem.total / (1024 ** 3), 2)
        info["available_gb"] = round(mem.available / (1024 ** 3), 2)
        info["used_gb"] = round(mem.used / (1024 ** 3), 2)
        info["free_gb"] = round(mem.free / (1024 ** 3), 2)
        
        # More detailed info from /proc/meminfo on Linux
        if platform.system() == "Linux":
            try:
                with open("/proc/meminfo") as f:
                    meminfo = f.read()
                
                detailed_info = {}
                for line in meminfo.split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        detailed_info[key.strip()] = value.strip()
                
                info["details"] = detailed_info
            except Exception as e:
                logger.error(f"Error reading /proc/meminfo: {str(e)}")
        
        return info
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """
        Get GPU information if available.
        
        Returns:
            Dictionary with GPU information
        """
        info = {
            "gpu_found": False,
            "devices": []
        }
        
        # Try to detect NVIDIA GPUs using nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free,memory.used,temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True
            )
            
            if result.stdout.strip():
                info["gpu_found"] = True
                info["type"] = "nvidia"
                
                # Parse nvidia-smi output
                for i, line in enumerate(result.stdout.strip().split("\n")):
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) >= 7:
                        gpu_info = {
                            "index": i,
                            "name": parts[0],
                            "driver_version": parts[1],
                            "memory_total_mb": float(parts[2]),
                            "memory_free_mb": float(parts[3]),
                            "memory_used_mb": float(parts[4]),
                            "temperature_c": float(parts[5]),
                            "utilization_percent": float(parts[6])
                        }
                        info["devices"].append(gpu_info)
        except (subprocess.SubprocessError, FileNotFoundError):
            # nvidia-smi not found or failed, try looking for AMD GPUs
            pass
        
        # Check for AMD GPUs if NVIDIA not found
        if not info["gpu_found"]:
            try:
                result = subprocess.run(["rocm-smi", "--showproductname"], capture_output=True, text=True, check=True)
                
                if "GPU" in result.stdout:
                    info["gpu_found"] = True
                    info["type"] = "amd"
                    
                    # Try to parse rocm-smi output (simplified)
                    for line in result.stdout.strip().split("\n"):
                        if ":" in line and "GPU" in line:
                            parts = line.split(":", 1)
                            if len(parts) >= 2:
                                info["devices"].append({
                                    "name": parts[1].strip()
                                })
            except (subprocess.SubprocessError, FileNotFoundError):
                # rocm-smi not found or failed, try looking for Intel GPUs
                pass
        
        # Check for Intel GPUs if neither NVIDIA nor AMD found
        if not info["gpu_found"] and platform.system() == "Linux":
            try:
                result = subprocess.run(["lspci", "-v"], capture_output=True, text=True, check=True)
                
                for line in result.stdout.lower().split("\n"):
                    if "vga" in line or "3d controller" in line:
                        if "intel" in line and "graphics" in line:
                            info["gpu_found"] = True
                            info["type"] = "intel"
                            info["devices"].append({
                                "name": line.split(":", 2)[-1].strip() if ":" in line else line
                            })
            except (subprocess.SubprocessError, FileNotFoundError):
                pass
        
        return info
    
    def get_disk_info(self) -> Dict[str, Any]:
        """
        Get disk space and I/O information.
        
        Returns:
            Dictionary with disk information
        """
        partitions = []
        
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                partition_info = {
                    "device": partition.device,
                    "mountpoint": partition.mountpoint,
                    "fstype": partition.fstype,
                    "opts": partition.opts,
                    "total": usage.total,
                    "used": usage.used,
                    "free": usage.free,
                    "percent": usage.percent,
                    "total_gb": round(usage.total / (1024 ** 3), 2),
                    "used_gb": round(usage.used / (1024 ** 3), 2),
                    "free_gb": round(usage.free / (1024 ** 3), 2),
                }
                partitions.append(partition_info)
            except (PermissionError, FileNotFoundError):
                # Some mount points might not be accessible
                continue
        
        # Get disk I/O statistics
        io_stats = psutil.disk_io_counters(perdisk=True)
        io_info = {}
        
        for disk, stats in io_stats.items():
            io_info[disk] = stats._asdict()
        
        return {
            "partitions": partitions,
            "io_stats": io_info
        }
    
    def get_network_info(self) -> Dict[str, Any]:
        """
        Get network interface and connectivity information.
        
        Returns:
            Dictionary with network information
        """
        interfaces = []
        
        # Get network interfaces
        for name, addrs in psutil.net_if_addrs().items():
            interface = {"name": name, "addresses": []}
            
            for addr in addrs:
                address_info = {
                    "family": str(addr.family),
                    "address": addr.address,
                    "netmask": addr.netmask,
                    "broadcast": addr.broadcast,
                }
                interface["addresses"].append(address_info)
            
            # Get interface statistics
            if name in psutil.net_if_stats():
                stats = psutil.net_if_stats()[name]
                interface["stats"] = {
                    "isup": stats.isup,
                    "duplex": stats.duplex,
                    "speed": stats.speed,
                    "mtu": stats.mtu,
                }
            
            interfaces.append(interface)
        
        # Get network I/O statistics
        io_stats = psutil.net_io_counters(pernic=True)
        io_info = {}
        
        for nic, stats in io_stats.items():
            io_info[nic] = stats._asdict()
        
        # Test internet connectivity
        connected = False
        latency = None
        
        try:
            # Try to ping Google's DNS to check connectivity (with 1 second timeout)
            result = subprocess.run(
                ["ping", "-c", "1", "-W", "1", "8.8.8.8"],
                capture_output=True, text=True
            )
            connected = result.returncode == 0
            
            # Extract ping latency if connected
            if connected:
                for line in result.stdout.split("\n"):
                    if "time=" in line:
                        time_part = line.split("time=")[1].split()[0]
                        try:
                            latency = float(time_part)
                        except ValueError:
                            pass
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        
        return {
            "interfaces": interfaces,
            "io_stats": io_info,
            "internet_connected": connected,
            "ping_latency_ms": latency
        }
    
    def get_docker_info(self) -> Dict[str, Any]:
        """
        Get Docker availability and capability information.
        
        Returns:
            Dictionary with Docker information
        """
        info = {
            "available": False,
            "version": None,
            "running_containers": 0,
            "images": 0,
        }
        
        try:
            client = docker.from_env()
            version_info = client.version()
            
            info["available"] = True
            info["version"] = version_info.get("Version")
            info["api_version"] = version_info.get("ApiVersion")
            info["running_containers"] = len(client.containers.list())
            info["images"] = len(client.images.list())
            
            # Get Docker system information
            system_info = client.info()
            info["os"] = system_info.get("OperatingSystem")
            info["architecture"] = system_info.get("Architecture")
            info["cpus"] = system_info.get("NCPU")
            info["memory"] = system_info.get("MemTotal")
            
            # Check if we have GPU access in Docker
            info["gpu_support"] = False
            
            if "Runtimes" in system_info and "nvidia" in system_info["Runtimes"]:
                info["gpu_support"] = True
            
        except Exception as e:
            logger.info(f"Docker not available: {str(e)}")
        
        return info
    
    def determine_capabilities(self) -> Dict[str, Any]:
        """
        Determine AI capabilities based on hardware audit results.
        
        Returns:
            Dictionary with capability assessments
        """
        capabilities = {
            "llm_inference": {
                "supported": False,
                "recommended_models": [],
                "limitations": []
            },
            "llm_fine_tuning": {
                "supported": False,
                "limitations": []
            },
            "image_generation": {
                "supported": False,
                "recommended_models": [],
                "limitations": []
            },
            "audio_processing": {
                "supported": True,
                "limitations": []
            },
            "video_processing": {
                "supported": False,
                "limitations": []
            },
            "containerization": {
                "supported": False,
                "limitations": []
            }
        }
        
        # Check CPU capability for basic LLM inference
        if self.cpu_info.get("logical_cores", 0) >= 4:
            capabilities["llm_inference"]["supported"] = True
            capabilities["llm_inference"]["recommended_models"].append("GPT-3.5-Turbo (API)")
        else:
            capabilities["llm_inference"]["limitations"].append("Insufficient CPU cores")
        
        # Memory requirements for different models
        memory_gb = self.memory_info.get("total_gb", 0)
        
        if memory_gb >= 8:
            capabilities["llm_inference"]["recommended_models"].append("Llama-2-7B")
        
        if memory_gb >= 16:
            capabilities["llm_inference"]["recommended_models"].append("Llama-2-13B")
            capabilities["image_generation"]["supported"] = True
            capabilities["image_generation"]["recommended_models"].append("Stable Diffusion")
        
        if memory_gb < 8:
            capabilities["llm_inference"]["limitations"].append("Limited memory for local model inference")
            capabilities["image_generation"]["limitations"].append("Insufficient memory for image generation")
        
        # GPU capabilities
        gpu_info = self.gpu_info
        if gpu_info.get("gpu_found", False):
            gpu_type = gpu_info.get("type", "")
            devices = gpu_info.get("devices", [])
            
            if gpu_type == "nvidia" and devices:
                # Check if we have enough VRAM for more capable models
                for device in devices:
                    memory_gb = device.get("memory_total_mb", 0) / 1024
                    
                    if memory_gb >= 8:
                        if "GPT-3.5-Turbo (Local)" not in capabilities["llm_inference"]["recommended_models"]:
                            capabilities["llm_inference"]["recommended_models"].append("GPT-3.5-Turbo (Local)")
                    
                    if memory_gb >= 16:
                        if "llama-2-13B" not in capabilities["llm_inference"]["recommended_models"]:
                            capabilities["llm_inference"]["recommended_models"].append("Llama-2-13B")
                        capabilities["llm_fine_tuning"]["supported"] = True
                    
                    if memory_gb >= 24:
                        capabilities["video_processing"]["supported"] = True
        else:
            capabilities["llm_inference"]["limitations"].append("No GPU found for acceleration")
            capabilities["image_generation"]["limitations"].append("No GPU found for acceleration")
            capabilities["video_processing"]["limitations"].append("No GPU found for processing")
        
        # Containerization support
        if self.docker_info.get("available", False):
            capabilities["containerization"]["supported"] = True
        else:
            capabilities["containerization"]["limitations"].append("Docker not available")
        
        return capabilities

def generate_hardware_report() -> Dict[str, Any]:
    """
    Generate a complete hardware audit report.
    
    Returns:
        Dictionary with complete hardware audit results
    """
    auditor = HardwareAudit()
    return auditor.run_full_audit()

def main():
    """Run a hardware audit and print the results."""
    logging.basicConfig(level=logging.INFO)
    report = generate_hardware_report()
    
    # Pretty print the report
    print(json.dumps(report, indent=4))
    
    # Print capability summary
    print("\n=== Capability Summary ===")
    for cap_name, cap_info in report["capabilities"].items():
        supported = "✅" if cap_info["supported"] else "❌"
        print(f"{cap_name}: {supported}")
        
        if cap_info["supported"] and "recommended_models" in cap_info:
            models = ", ".join(cap_info["recommended_models"])
            print(f"  Recommended models: {models}")
        
        if cap_info["limitations"]:
            limitations = ", ".join(cap_info["limitations"])
            print(f"  Limitations: {limitations}")

if __name__ == "__main__":
    main()
