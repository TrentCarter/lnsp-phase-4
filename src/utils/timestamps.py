#!/usr/bin/env python3
"""Timestamp utilities for LNSP CPESH caching system."""

import time
from datetime import datetime, timezone
from typing import Optional, Tuple


def get_iso_timestamp() -> str:
    """
    Get current timestamp in ISO8601 format with timezone.

    Returns:
        ISO8601 formatted timestamp string (e.g., "2025-09-25T07:27:21.123456+00:00")
    """
    return datetime.now(timezone.utc).isoformat()


def parse_iso_timestamp(timestamp_str: str) -> Optional[datetime]:
    """
    Parse ISO8601 timestamp string to datetime object.

    Args:
        timestamp_str: ISO8601 formatted timestamp string

    Returns:
        datetime object if parsing succeeds, None otherwise
    """
    if not timestamp_str:
        return None

    try:
        # Try parsing with timezone info first
        return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    except ValueError:
        try:
            # Fallback: try parsing without timezone, assume UTC
            dt = datetime.fromisoformat(timestamp_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            return None


def get_unix_timestamp() -> float:
    """
    Get current Unix timestamp (seconds since epoch).

    Returns:
        Unix timestamp as float
    """
    return time.time()


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string (e.g., "1h 30m 45s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


def timestamp_to_age(timestamp_str: str) -> Optional[str]:
    """
    Convert timestamp to human-readable age string.

    Args:
        timestamp_str: ISO8601 formatted timestamp string

    Returns:
        Age string (e.g., "2 hours ago") or None if parsing fails
    """
    if not timestamp_str:
        return None

    dt = parse_iso_timestamp(timestamp_str)
    if dt is None:
        return None

    now = datetime.now(timezone.utc)
    delta = now - dt

    seconds = delta.total_seconds()
    if seconds < 60:
        return "just now"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif seconds < 86400:
        hours = int(seconds // 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    else:
        days = int(seconds // 86400)
        return f"{days} day{'s' if days != 1 else ''} ago"


def is_timestamp_expired(timestamp_str: str, max_age_seconds: float) -> bool:
    """
    Check if a timestamp is older than the specified maximum age.

    Args:
        timestamp_str: ISO8601 formatted timestamp string
        max_age_seconds: Maximum allowed age in seconds

    Returns:
        True if timestamp is expired (older than max_age_seconds), False otherwise
    """
    if not timestamp_str:
        return True

    dt = parse_iso_timestamp(timestamp_str)
    if dt is None:
        return True

    now = datetime.now(timezone.utc)
    age = (now - dt).total_seconds()

    return age > max_age_seconds


def validate_timestamp_format(timestamp_str: str) -> bool:
    """
    Validate that a string is a properly formatted ISO8601 timestamp.

    Args:
        timestamp_str: String to validate

    Returns:
        True if valid ISO8601 format, False otherwise
    """
    return parse_iso_timestamp(timestamp_str) is not None


def migrate_legacy_cache_entry(entry: dict) -> dict:
    """
    Migrate legacy cache entry to include timestamp fields if missing.

    Args:
        entry: Cache entry dictionary

    Returns:
        Updated cache entry with timestamp fields
    """
    if not isinstance(entry, dict):
        return entry

    cpesh = entry.get('cpesh')
    if not isinstance(cpesh, dict):
        return entry

    # Core field aliases for backward compatibility
    if 'concept_text' not in cpesh and 'concept' in cpesh:
        cpesh['concept_text'] = cpesh.get('concept')
    if 'probe_question' not in cpesh and 'probe' in cpesh:
        cpesh['probe_question'] = cpesh.get('probe')
    if 'expected_answer' not in cpesh and 'expected' in cpesh:
        cpesh['expected_answer'] = cpesh.get('expected')

    # Ensure optional metadata is present
    cpesh.setdefault('mission_text', None)
    cpesh.setdefault('dataset_source', None)
    cpesh.setdefault('content_type', None)
    cpesh.setdefault('chunk_position', None)
    cpesh.setdefault('relations_text', cpesh.get('relations_text') or [])
    cpesh.setdefault('tmd_bits', cpesh.get('tmd_bits'))
    cpesh.setdefault('tmd_lane', cpesh.get('tmd_lane'))
    cpesh.setdefault('lane_index', cpesh.get('lane_index'))
    cpesh.setdefault('quality', cpesh.get('quality'))
    cpesh.setdefault('echo_score', cpesh.get('echo_score'))
    cpesh.setdefault('insufficient_evidence', cpesh.get('insufficient_evidence', False))

    # Set created_at if missing
    if 'created_at' not in cpesh:
        cpesh['created_at'] = get_iso_timestamp()

    # Set last_accessed if missing (use created_at as fallback)
    if 'last_accessed' not in cpesh:
        cpesh['last_accessed'] = cpesh['created_at']

    # Add access_count if missing
    access_count = entry.get('access_count')
    entry['access_count'] = int(access_count) if isinstance(access_count, int) else 1
    if entry['access_count'] <= 0:
        entry['access_count'] = 1

    entry['quality'] = cpesh.get('quality')
    entry['cosine'] = cpesh.get('soft_sim')

    return entry


def update_cache_entry_access(entry: dict) -> dict:
    """
    Update cache entry access timestamp and increment counter.

    Args:
        entry: Cache entry dictionary

    Returns:
        Updated cache entry with new last_accessed timestamp and incremented access_count
    """
    if not isinstance(entry, dict):
        return entry

    now = get_iso_timestamp()

    cpesh = entry.get('cpesh')
    if isinstance(cpesh, dict):
        cpesh['last_accessed'] = now

    count = entry.get('access_count', 0)
    entry['access_count'] = int(count) + 1 if isinstance(count, (int, float)) else 1

    if isinstance(cpesh, dict):
        cpesh['access_count'] = entry['access_count']

    return entry


def migrate_cpesh_record(record: dict) -> dict:
    """Backfill CPESH record dictionaries with authoritative fields."""
    if not isinstance(record, dict):
        return record

    now = get_iso_timestamp()
    record.setdefault('cpe_id', record.get('id'))
    record.setdefault('concept_text', record.get('concept'))
    record.setdefault('probe_question', record.get('probe'))
    record.setdefault('expected_answer', record.get('expected'))
    record.setdefault('mission_text', record.get('mission_text'))
    record.setdefault('source_chunk', record.get('source_chunk'))
    record.setdefault('dataset_source', record.get('dataset_source'))
    record.setdefault('content_type', record.get('content_type'))
    record.setdefault('chunk_position', record.get('chunk_position'))
    record.setdefault('relations_text', record.get('relations_text') or [])
    record.setdefault('tmd_bits', record.get('tmd_bits'))
    record.setdefault('tmd_lane', record.get('tmd_lane'))
    record.setdefault('lane_index', record.get('lane_index'))
    record.setdefault('quality', record.get('quality'))
    record.setdefault('echo_score', record.get('echo_score'))
    record.setdefault('insufficient_evidence', record.get('insufficient_evidence', False))

    created = record.get('created_at') or now
    record['created_at'] = created
    record['last_accessed'] = record.get('last_accessed') or created

    count = record.get('access_count', 0)
    record['access_count'] = int(count) if isinstance(count, (int, float)) else 0
    if record['access_count'] < 0:
        record['access_count'] = 0

    return record
