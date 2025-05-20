#!/usr/bin/env python3
"""
CLI Output Utilities

This module provides utilities for formatting CLI output with colors and styles.
"""

import sys
from typing import Any, Optional

# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    
    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"
    
    # Bright foreground colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

def print_header(text: str, color: str = Colors.BLUE) -> None:
    """
    Print a header with the specified color.
    
    Args:
        text: Header text
        color: ANSI color code
    """
    print(f"\n{color}{Colors.BOLD}=== {text} ==={Colors.RESET}\n")

def print_success(text: str) -> None:
    """
    Print a success message.
    
    Args:
        text: Success message
    """
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")

def print_error(text: str) -> None:
    """
    Print an error message.
    
    Args:
        text: Error message
    """
    print(f"{Colors.RED}✗ {text}{Colors.RESET}", file=sys.stderr)

def print_warning(text: str) -> None:
    """
    Print a warning message.
    
    Args:
        text: Warning message
    """
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")

def print_info(text: str) -> None:
    """
    Print an info message.
    
    Args:
        text: Info message
    """
    print(f"{Colors.CYAN}ℹ {text}{Colors.RESET}")

def print_debug(text: str) -> None:
    """
    Print a debug message.
    
    Args:
        text: Debug message
    """
    print(f"{Colors.BRIGHT_BLACK}➤ {text}{Colors.RESET}")

def print_progress(text: str) -> None:
    """
    Print a progress message.
    
    Args:
        text: Progress message
    """
    print(f"{Colors.MAGENTA}⏳ {text}{Colors.RESET}")

def print_task(task_id: str, task_type: str, status: str) -> None:
    """
    Print a task status.
    
    Args:
        task_id: Task ID
        task_type: Task type
        status: Task status
    """
    status_color = Colors.GREEN if status == "success" else Colors.RED if status == "failure" else Colors.YELLOW
    
    print(f"{Colors.CYAN}{task_id}{Colors.RESET} [{Colors.BLUE}{task_type}{Colors.RESET}] - {status_color}{status}{Colors.RESET}")

def format_json(data: Any) -> str:
    """
    Format JSON data with syntax highlighting.
    
    Args:
        data: JSON data
        
    Returns:
        Formatted JSON string
    """
    import json
    
    formatted = json.dumps(data, indent=2)
    
    # Add some basic syntax highlighting
    formatted = formatted.replace('"', f'{Colors.GREEN}"') \
                        .replace('": ', f'"{Colors.RESET}: ') \
                        .replace('{', f'{Colors.YELLOW}{{{Colors.RESET}') \
                        .replace('}', f'{Colors.YELLOW}}}{Colors.RESET}') \
                        .replace('[', f'{Colors.YELLOW}[{Colors.RESET}') \
                        .replace(']', f'{Colors.YELLOW}]{Colors.RESET}') \
                        .replace('true', f'{Colors.MAGENTA}true{Colors.RESET}') \
                        .replace('false', f'{Colors.MAGENTA}false{Colors.RESET}') \
                        .replace('null', f'{Colors.MAGENTA}null{Colors.RESET}')
    
    return formatted

def clear_screen() -> None:
    """Clear the terminal screen."""
    print("\033c", end="")

def print_table(
    headers: list, 
    rows: list, 
    header_color: str = Colors.BLUE, 
    alt_row_color: Optional[str] = Colors.BRIGHT_BLACK
) -> None:
    """
    Print a table with the specified headers and rows.
    
    Args:
        headers: List of column headers
        rows: List of row data (each row is a list of cell values)
        header_color: ANSI color code for headers
        alt_row_color: ANSI color code for alternate rows (None for no alternating)
    """
    # Calculate column widths
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(str(cell)))
    
    # Print headers
    header_row = " | ".join(f"{h:{w}}" for h, w in zip(headers, widths))
    print(f"{header_color}{Colors.BOLD}{header_row}{Colors.RESET}")
    
    # Print separator
    separator = "-+-".join("-" * w for w in widths)
    print(separator)
    
    # Print rows
    for i, row in enumerate(rows):
        row_color = alt_row_color if alt_row_color and i % 2 == 1 else ""
        row_str = " | ".join(f"{str(cell):{w}}" for cell, w in zip(row, widths))
        print(f"{row_color}{row_str}{Colors.RESET}")

# Example usage
if __name__ == "__main__":
    clear_screen()
    
    print_header("Autonomous AI Architect CLI")
    
    print_info("Starting the system...")
    print_progress("Initializing components...")
    print_success("System initialized successfully")
    
    print_warning("Low disk space available")
    print_error("Failed to connect to Redis")
    
    print_header("Task Results")
    
    print_task("task-001", "hardware_analysis", "success")
    print_task("task-002", "architecture", "in_progress")
    print_task("task-003", "code_execution", "failure")
    
    print_header("Hardware Analysis")
    
    print_table(
        ["Component", "Status", "Details"],
        [
            ["CPU", "OK", "Intel Core i7-9700K @ 3.60GHz"],
            ["Memory", "Warning", "16GB (80% used)"],
            ["GPU", "Not Found", "N/A"],
            ["Disk", "OK", "512GB SSD (40% used)"]
        ]
    )
    
    print_header("JSON Output")
    
    example_data = {
        "status": "success",
        "results": [
            {"id": 1, "name": "Example 1", "active": True},
            {"id": 2, "name": "Example 2", "active": False},
            {"id": 3, "name": "Example 3", "active": None}
        ]
    }
    
    print(format_json(example_data))
