#!/usr/bin/env python3
"""
Datetime utility functions for Milvus timestamp conversion
"""

from datetime import datetime
from typing import Union
import logging

logger = logging.getLogger(__name__)

def datetime_to_timestamp(dt: Union[datetime, str, None]) -> int:
    """
    Convert datetime to Unix timestamp in milliseconds for Milvus storage
    
    Args:
        dt: datetime object, ISO format string, or None
        
    Returns:
        Unix timestamp in milliseconds
    """
    if dt is None:
        return int(datetime.now().timestamp() * 1000)
    
    if isinstance(dt, str):
        # Handle ISO format datetime strings from SQL
        try:
            # Remove 'T' and parse as datetime
            dt_clean = dt.replace('T', ' ')
            parsed_dt = datetime.fromisoformat(dt_clean)
            return int(parsed_dt.timestamp() * 1000)
        except ValueError:
            # Fallback to current time if parsing fails
            logger.warning(f"Failed to parse datetime string '{dt}', using current time")
            return int(datetime.now().timestamp() * 1000)
    
    if hasattr(dt, 'timestamp'):
        # It's a datetime object
        return int(dt.timestamp() * 1000)
    
    # Fallback for other types
    logger.warning(f"Unexpected datetime type {type(dt)}, using current time")
    return int(datetime.now().timestamp() * 1000)

def timestamp_to_datetime(timestamp: int) -> datetime:
    """
    Convert Unix timestamp in milliseconds to datetime
    
    Args:
        timestamp: Unix timestamp in milliseconds
        
    Returns:
        datetime object
    """
    return datetime.fromtimestamp(timestamp / 1000.0)
