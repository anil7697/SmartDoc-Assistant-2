"""Utility functions and logging setup for Document Q&A application."""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from config import settings


def setup_logging():
    """Configure logging for the application."""
    # Remove default logger
    logger.remove()
    
    # Add console logging
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
    )
    
    # Add file logging
    log_file_path = Path(settings.log_file)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        log_file_path,
        level=settings.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
    )
    
    return logger


def validate_api_keys() -> Dict[str, bool]:
    """Validate that required API keys are available."""
    validation_results = {
        "google": bool(settings.google_api_key),
    }

    return validation_results


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    import re
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(' .')
    # Limit length
    if len(sanitized) > 255:
        name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
        sanitized = name[:255-len(ext)-1] + ('.' + ext if ext else '')
    
    return sanitized


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


# Initialize logging when module is imported
setup_logging()
