#!/usr/bin/env python3
"""
Flat Log Communication Logger for PAS
Logs all parent-child communications, status updates, and commands.

Format: timestamp|from|to|type|message|llm_model|run_id|status|progress|metadata

Dual output:
- Flat .txt files (artifacts/logs/pas_comms_<date>.txt)
- SQLite action_logs table (artifacts/registry/registry.db)

See: docs/FLAT_LOG_FORMAT.md
"""
import os
import json
import pathlib
import sqlite3
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Optional, Dict, Any
from urllib.parse import quote
from enum import Enum


class MessageType(str, Enum):
    """Message type enum"""
    CMD = "CMD"              # Command from parent to child
    STATUS = "STATUS"        # Status update from child to parent
    HEARTBEAT = "HEARTBEAT"  # Periodic health check
    RESPONSE = "RESPONSE"    # Response to a command


def get_llm_code(llm_model: Optional[str]) -> str:
    """
    Extract 6-character LLM code from model string.

    Examples:
        "anthropic/claude-4.5-sonnet" -> "CLD450"
        "anthropic/claude-3-opus" -> "CLD30O"
        "openai/gpt-4.5-turbo" -> "GPT450"
        "google/gemini-2.5-flash" -> "GMI250"
        "ollama/qwen2.5-coder:7b-instruct" -> "QWE250"
        None or "-" -> "------"
    """
    if not llm_model or llm_model == "-":
        return "------"

    llm_lower = llm_model.lower()

    # Claude models
    if "claude" in llm_lower:
        if "4.5" in llm_lower or "sonnet-4" in llm_lower:
            return "CLD450"
        elif "3.7" in llm_lower:
            return "CLD370"
        elif "3.5" in llm_lower:
            return "CLD350"
        elif "3" in llm_lower:
            if "opus" in llm_lower:
                return "CLD30O"
            elif "sonnet" in llm_lower:
                return "CLD30S"
            elif "haiku" in llm_lower:
                return "CLD30H"
            return "CLD300"
        return "CLAUDE"

    # GPT models
    if "gpt" in llm_lower:
        if "4.5" in llm_lower or "4-5" in llm_lower:
            return "GPT450"
        elif "3.5" in llm_lower or "3-5" in llm_lower:
            return "GPT350"
        elif "5" in llm_lower and "3.5" not in llm_lower and "4.5" not in llm_lower:
            return "GPT500"
        elif "4" in llm_lower:
            return "GPT400"
        elif "3" in llm_lower:
            return "GPT300"
        return "GPTXXX"

    # Gemini models
    if "gemini" in llm_lower:
        if "2.5" in llm_lower:
            if "flash" in llm_lower:
                return "GMI250"
            return "GM250P"
        elif "2" in llm_lower:
            return "GMI200"
        elif "1.5" in llm_lower:
            return "GMI150"
        return "GEMINI"

    # Qwen models
    if "qwen" in llm_lower:
        if "2.5" in llm_lower:
            return "QWE250"
        elif "2" in llm_lower:
            return "QWE200"
        return "QWENXX"

    # Llama models
    if "llama" in llm_lower:
        if "3.1" in llm_lower or "31" in llm_lower:
            return "LMA310"
        elif "3" in llm_lower:
            return "LMA300"
        elif "2" in llm_lower:
            return "LMA200"
        return "LLAMAX"

    # Unknown - return first 6 chars of model name (uppercase)
    model_name = llm_model.split("/")[-1].split(":")[0][:6].upper()
    return model_name.ljust(6, "X")


class CommsLogger:
    """
    Flat file logger for PAS parent-child communications.

    Features:
    - Append-only writes (thread-safe)
    - Daily rotation
    - Per-run and global logs
    - Automatic escaping of pipes/newlines
    - URL-encoded metadata JSON
    """

    def __init__(
        self,
        log_dir: str = "artifacts/logs",
        db_path: str = "artifacts/registry/registry.db",
        buffer_size: int = 4096,
        flush_interval_s: int = 5,
        max_line_bytes: int = 65536
    ):
        self.log_dir = pathlib.Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = pathlib.Path(db_path)
        self.buffer_size = buffer_size
        self.flush_interval_s = flush_interval_s
        self.max_line_bytes = max_line_bytes

        # Use EST timezone for all timestamps
        self.est_tz = ZoneInfo("America/New_York")

        # Global log file (daily rotation)
        self._global_log_path = self._get_daily_log_path()
        self._global_log_file = None
        self._last_flush = datetime.now(self.est_tz)

    def _get_daily_log_path(self) -> pathlib.Path:
        """Get log file path for today (EST timezone)"""
        date_str = datetime.now(self.est_tz).strftime("%Y-%m-%d")
        return self.log_dir / f"pas_comms_{date_str}.txt"

    def _escape(self, s: str) -> str:
        """Escape pipe and newline characters"""
        if not s:
            return "-"
        return s.replace("|", "\\|").replace("\n", "\\n")

    def _format_metadata(self, metadata: Optional[Dict[str, Any]]) -> str:
        """Format metadata as URL-encoded JSON"""
        if not metadata:
            return "-"
        try:
            json_str = json.dumps(metadata, separators=(',', ':'))
            return quote(json_str)
        except Exception:
            return "-"

    def _format_line(
        self,
        from_agent: str,
        to_agent: str,
        msg_type: MessageType,
        message: str,
        llm_model: Optional[str] = None,
        run_id: Optional[str] = None,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format a log line according to spec.

        Format: timestamp|from|to|type|llm_code|message|llm_model|run_id|status|progress|metadata
        """
        timestamp = datetime.now(self.est_tz).isoformat(timespec='milliseconds')

        # Get LLM code (5-char identifier)
        llm_code = get_llm_code(llm_model)

        # Format fields (use '-' for missing optional fields)
        fields = [
            timestamp,
            self._escape(from_agent),
            self._escape(to_agent),
            msg_type.value,
            llm_code,  # NEW: 5-char LLM code
            self._escape(message),
            self._escape(llm_model or "-"),
            self._escape(run_id or "-"),
            self._escape(status or "-"),
            f"{progress:.2f}" if progress is not None else "-",
            self._format_metadata(metadata)
        ]

        line = "|".join(fields) + "\n"

        # Truncate if too long
        if len(line.encode('utf-8')) > self.max_line_bytes:
            truncated_msg = message[:1000] + "...[truncated]"
            fields[5] = self._escape(truncated_msg)  # Updated index (was 4, now 5)
            fields[10] = "-"  # Drop metadata if line too long (was 9, now 10)
            line = "|".join(fields) + "\n"

        return line

    def _write_to_file(self, log_path: pathlib.Path, line: str):
        """Append line to log file"""
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception as e:
            # Fallback: print to stderr if file write fails
            import sys
            print(f"[CommsLogger] Failed to write to {log_path}: {e}", file=sys.stderr)
            print(line.rstrip(), file=sys.stderr)

    def _write_to_db(
        self,
        from_agent: str,
        to_agent: str,
        msg_type: MessageType,
        message: str,
        run_id: Optional[str] = None,
        status: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parent_log_id: Optional[int] = None
    ):
        """Write to action_logs database table with parent-child tracking"""
        if not self.db_path.exists():
            return  # Silently skip if database doesn't exist

        try:
            conn = sqlite3.connect(str(self.db_path), timeout=5.0)
            cursor = conn.cursor()

            timestamp = datetime.now(self.est_tz).isoformat(timespec='milliseconds')
            task_id = run_id if run_id and run_id != "-" else "unknown"

            # Map MessageType to action_type/action_name
            action_type_map = {
                MessageType.CMD: ("delegate", message),
                MessageType.STATUS: ("status_update", message),
                MessageType.HEARTBEAT: ("heartbeat", message),
                MessageType.RESPONSE: ("response", message)
            }
            action_type, action_name = action_type_map.get(msg_type, ("unknown", message))

            # Format action_data JSON
            action_data = {
                "message": message,
                "status": status,
                "metadata": metadata or {}
            }

            cursor.execute("""
                INSERT INTO action_logs (
                    task_id, parent_log_id, timestamp,
                    from_agent, to_agent,
                    action_type, action_name, action_data,
                    status, tier_from, tier_to
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task_id,
                parent_log_id,  # parent_log_id passed from caller
                timestamp,
                from_agent,
                to_agent,
                action_type,
                action_name,
                json.dumps(action_data),
                status or "unknown",
                None,  # tier_from (could be inferred from agent name)
                None   # tier_to (could be inferred from agent name)
            ))

            # Get the new log_id to return for child tracking
            log_id = cursor.lastrowid

            conn.commit()
            conn.close()

            return log_id  # Return log_id for parent-child tracking

        except Exception as e:
            # Silently skip database errors (flat files are primary)
            import sys
            print(f"[CommsLogger] DB write error: {e}", file=sys.stderr)
            return None

    def log(
        self,
        from_agent: str,
        to_agent: str,
        msg_type: MessageType,
        message: str,
        llm_model: Optional[str] = None,
        run_id: Optional[str] = None,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parent_log_id: Optional[int] = None
    ):
        """
        Log a communication event.

        Args:
            from_agent: Source agent/service name
            to_agent: Destination agent/service name
            msg_type: Message type (CMD, STATUS, HEARTBEAT, RESPONSE)
            message: Human-readable message
            llm_model: LLM model name (optional)
            run_id: Run identifier (optional)
            status: Agent status (optional)
            progress: Progress ratio 0.0-1.0 (optional)
            metadata: Additional key-value data (optional)
            parent_log_id: Parent log entry ID for hierarchy tracking (optional)

        Returns:
            log_id: Database log ID for parent-child tracking (or None if DB write failed)

        Example:
            # Log parent action
            parent_id = logger.log(
                from_agent="PAS Root",
                to_agent="Aider-LCO",
                msg_type=MessageType.CMD,
                message="Execute Prime Directive",
                llm_model="ollama/qwen2.5-coder:7b-instruct",
                run_id="abc123-def456",
                status="queued",
                progress=0.0
            )

            # Log child action
            logger.log(
                from_agent="Aider-LCO",
                to_agent="PAS Root",
                msg_type=MessageType.RESPONSE,
                message="Task completed",
                run_id="abc123-def456",
                parent_log_id=parent_id  # Link to parent
            )
        """
        line = self._format_line(
            from_agent=from_agent,
            to_agent=to_agent,
            msg_type=msg_type,
            message=message,
            llm_model=llm_model,
            run_id=run_id,
            status=status,
            progress=progress,
            metadata=metadata
        )

        # Write to global log
        current_log_path = self._get_daily_log_path()
        self._write_to_file(current_log_path, line)

        # Write to per-run log (if run_id provided)
        if run_id and run_id != "-":
            run_log_dir = pathlib.Path(f"artifacts/runs/{run_id}")
            run_log_dir.mkdir(parents=True, exist_ok=True)
            run_log_path = run_log_dir / "comms.txt"
            self._write_to_file(run_log_path, line)

        # Write to action_logs database (for HMI visualization)
        log_id = self._write_to_db(
            from_agent=from_agent,
            to_agent=to_agent,
            msg_type=msg_type,
            message=message,
            run_id=run_id,
            status=status,
            metadata=metadata,
            parent_log_id=parent_log_id
        )

        return log_id  # Return log_id for parent-child tracking

    def log_cmd(self, from_agent: str, to_agent: str, message: str, run_id: Optional[str] = None, **kwargs):
        """Convenience: Log a command. Returns log_id for parent-child tracking."""
        return self.log(from_agent, to_agent, MessageType.CMD, message, run_id=run_id, **kwargs)

    def log_status(self, from_agent: str, to_agent: str, message: str, run_id: Optional[str] = None, status: Optional[str] = None, progress: Optional[float] = None, **kwargs):
        """Convenience: Log a status update. Returns log_id for parent-child tracking."""
        return self.log(from_agent, to_agent, MessageType.STATUS, message, run_id=run_id, status=status, progress=progress, **kwargs)

    def log_heartbeat(self, from_agent: str, to_agent: str, message: str, run_id: Optional[str] = None, status: Optional[str] = None, progress: Optional[float] = None, **kwargs):
        """Convenience: Log a heartbeat. Returns log_id for parent-child tracking."""
        return self.log(from_agent, to_agent, MessageType.HEARTBEAT, message, run_id=run_id, status=status, progress=progress, **kwargs)

    def log_response(self, from_agent: str, to_agent: str, message: str, run_id: Optional[str] = None, status: Optional[str] = None, **kwargs):
        """Convenience: Log a response. Returns log_id for parent-child tracking."""
        return self.log(from_agent, to_agent, MessageType.RESPONSE, message, run_id=run_id, status=status, **kwargs)

    def log_separator(self, label: str = "SEPARATOR", run_id: Optional[str] = None):
        """
        Log a visual separator (full-width line break).

        Args:
            label: Label for the separator (e.g., "START", "END", "SEPARATOR")
            run_id: Run identifier (optional)

        Example:
            logger.log_separator("START", run_id="abc123")
            # Output: ═══════════════════════════════════════ START ═══════════════════════════════════════
        """
        separator_line = f"{'═' * 40} {label} {'═' * 40}\n"

        # Write to global log
        current_log_path = self._get_daily_log_path()
        self._write_to_file(current_log_path, separator_line)

        # Write to per-run log (if run_id provided)
        if run_id and run_id != "-":
            run_log_dir = pathlib.Path(f"artifacts/runs/{run_id}")
            run_log_dir.mkdir(parents=True, exist_ok=True)
            run_log_path = run_log_dir / "comms.txt"
            self._write_to_file(run_log_path, separator_line)


# Global singleton instance
_logger_instance: Optional[CommsLogger] = None


def get_logger() -> CommsLogger:
    """Get global CommsLogger instance (singleton)"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = CommsLogger()
    return _logger_instance


# Convenience functions (use global logger)
def log_cmd(from_agent: str, to_agent: str, message: str, **kwargs):
    """Log a command (global logger)"""
    get_logger().log_cmd(from_agent, to_agent, message, **kwargs)


def log_status(from_agent: str, to_agent: str, message: str, **kwargs):
    """Log a status update (global logger)"""
    get_logger().log_status(from_agent, to_agent, message, **kwargs)


def log_heartbeat(from_agent: str, to_agent: str, message: str, **kwargs):
    """Log a heartbeat (global logger)"""
    get_logger().log_heartbeat(from_agent, to_agent, message, **kwargs)


def log_response(from_agent: str, to_agent: str, message: str, **kwargs):
    """Log a response (global logger)"""
    get_logger().log_response(from_agent, to_agent, message, **kwargs)


def log_separator(label: str = "SEPARATOR", **kwargs):
    """Log a visual separator (global logger)"""
    get_logger().log_separator(label, **kwargs)
