#!/usr/bin/env python3
"""
PAS Communication Log Parser and Viewer

Parse and query flat .txt communication logs.

Usage:
    # View all logs for today
    ./tools/parse_comms_log.py

    # View logs for specific run
    ./tools/parse_comms_log.py --run-id abc123-def456

    # View only commands
    ./tools/parse_comms_log.py --type CMD

    # View logs for specific agent
    ./tools/parse_comms_log.py --agent "Aider-LCO"

    # View logs with specific LLM
    ./tools/parse_comms_log.py --llm "claude"

    # Export to JSON
    ./tools/parse_comms_log.py --format json > logs.json

    # Watch logs in real-time (tail -f)
    ./tools/parse_comms_log.py --tail
"""
import argparse
import csv
import json
import sys
import pathlib
from datetime import datetime
from typing import Optional, List, Dict, Any
from urllib.parse import unquote
from dataclasses import dataclass


@dataclass
class LogEntry:
    """Parsed log entry"""
    timestamp: str
    from_agent: str
    to_agent: str
    msg_type: str
    llm_code: str  # NEW: 6-char LLM code (e.g., CLD450, GMI250, QWE250)
    message: str
    llm_model: str
    run_id: str
    status: str
    progress: str
    metadata: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp,
            "from": self.from_agent,
            "to": self.to_agent,
            "type": self.msg_type,
            "llm_code": self.llm_code if self.llm_code != "------" else None,
            "message": self.message,
            "llm_model": self.llm_model if self.llm_model != "-" else None,
            "run_id": self.run_id if self.run_id != "-" else None,
            "status": self.status if self.status != "-" else None,
            "progress": float(self.progress) if self.progress != "-" else None,
            "metadata": json.loads(unquote(self.metadata)) if self.metadata != "-" else None
        }

    def matches_filter(
        self,
        run_id: Optional[str] = None,
        msg_type: Optional[str] = None,
        agent: Optional[str] = None,
        llm: Optional[str] = None,
        status: Optional[str] = None
    ) -> bool:
        """Check if entry matches filter criteria"""
        if run_id and self.run_id != run_id:
            return False
        if msg_type and self.msg_type != msg_type:
            return False
        if agent and agent not in (self.from_agent, self.to_agent):
            return False
        if llm and llm not in self.llm_model:
            return False
        if status and self.status != status:
            return False
        return True


def parse_log_file(log_path: pathlib.Path) -> List[LogEntry]:
    """Parse log file and return entries (handles separator lines)"""
    entries = []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Detect separator lines (═══════════)
                if line.startswith("═" * 10):
                    # Print separator directly (colored)
                    print(f"\033[1;35m{line}\033[0m")  # Bold magenta
                    continue

                # Parse regular log entries
                try:
                    reader = csv.reader([line], delimiter="|")
                    row = next(reader)
                    # Support both old format (10 fields) and new format (11 fields with llm_code)
                    if len(row) == 10:
                        # Old format: insert default llm_code "------" after msg_type
                        row.insert(4, "------")
                    elif len(row) != 11:
                        continue  # Skip malformed lines
                    entry = LogEntry(*row)
                    entries.append(entry)
                except Exception:
                    continue  # Skip unparseable lines
    except FileNotFoundError:
        print(f"Error: Log file not found: {log_path}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Error parsing log file: {e}", file=sys.stderr)
        return []
    return entries


def format_entry_text(entry: LogEntry, color: bool = True) -> str:
    """Format entry as human-readable text"""
    # Color codes (ANSI)
    RESET = "\033[0m" if color else ""
    BOLD = "\033[1m" if color else ""
    CYAN = "\033[36m" if color else ""
    GREEN = "\033[32m" if color else ""
    YELLOW = "\033[33m" if color else ""
    RED = "\033[31m" if color else ""
    BLUE = "\033[34m" if color else ""
    MAGENTA = "\033[35m" if color else ""

    # Color by message type
    type_color = {
        "CMD": YELLOW,
        "STATUS": BLUE,
        "HEARTBEAT": GREEN,
        "RESPONSE": CYAN
    }.get(entry.msg_type, RESET)

    # Color by status
    status_color = {
        "completed": GREEN,
        "error": RED,
        "running": YELLOW,
        "queued": BLUE
    }.get(entry.status, RESET)

    # Format timestamp (remove microseconds for readability)
    ts = entry.timestamp.split(".")[0].replace("T", " ").replace("Z", "")

    # Format progress
    progress_str = ""
    if entry.progress != "-":
        progress_pct = int(float(entry.progress) * 100)
        progress_str = f" [{progress_pct}%]"

    # Format LLM code (6-char identifier between msg_type and from_agent)
    llm_code_str = ""
    if entry.llm_code and entry.llm_code != "------":
        llm_code_str = f"{MAGENTA}{entry.llm_code:6}{RESET}  "
    else:
        llm_code_str = f"        "  # 8 spaces (6 for code + 2 spacing)

    # Format LLM model (short form for end of line)
    llm_str = ""
    if entry.llm_model != "-":
        llm_short = entry.llm_model.split("/")[-1].split(":")[0]  # Extract model name only
        llm_str = f" {MAGENTA}[{llm_short}]{RESET}"

    # Format main line with LLM code between msg_type and from_agent
    line = (
        f"{CYAN}{ts}{RESET} "
        f"{type_color}{entry.msg_type:10}{RESET} "
        f"{llm_code_str}"  # NEW: LLM code between msg_type and from_agent
        f"{BOLD}{entry.from_agent:15}{RESET} → {BOLD}{entry.to_agent:15}{RESET} "
        f"{entry.message[:60]}"
    )

    # Add status if present
    if entry.status != "-":
        line += f" {status_color}[{entry.status}]{RESET}{progress_str}"

    # Add LLM model (keep for backward compatibility and detail)
    line += llm_str

    return line


def format_entry_json(entry: LogEntry) -> str:
    """Format entry as JSON"""
    return json.dumps(entry.to_dict(), indent=2)


def main():
    parser = argparse.ArgumentParser(description="Parse and query PAS communication logs")
    parser.add_argument("--log-file", type=str, help="Log file to parse (default: today's log)")
    parser.add_argument("--run-id", type=str, help="Filter by run ID")
    parser.add_argument("--type", type=str, choices=["CMD", "STATUS", "HEARTBEAT", "RESPONSE"], help="Filter by message type")
    parser.add_argument("--agent", type=str, help="Filter by agent name (from or to)")
    parser.add_argument("--llm", type=str, help="Filter by LLM model (substring match)")
    parser.add_argument("--status", type=str, help="Filter by status")
    parser.add_argument("--format", type=str, choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    parser.add_argument("--tail", action="store_true", help="Watch log file in real-time (like tail -f)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of entries shown")

    args = parser.parse_args()

    # Determine log file
    if args.log_file:
        log_path = pathlib.Path(args.log_file)
    else:
        # Default to today's log
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_path = pathlib.Path(f"artifacts/logs/pas_comms_{date_str}.txt")

    # Tail mode (real-time watching)
    if args.tail:
        print(f"Watching {log_path} (Ctrl+C to stop)...", file=sys.stderr)
        import time
        last_pos = 0
        try:
            while True:
                if log_path.exists():
                    with open(log_path, "r", encoding="utf-8") as f:
                        f.seek(last_pos)
                        new_lines = f.readlines()
                        last_pos = f.tell()

                        for line in new_lines:
                            line = line.strip()
                            if not line:
                                continue

                            # Detect separator lines
                            if line.startswith("═" * 10):
                                print(f"\033[1;35m{line}\033[0m")  # Bold magenta
                                continue

                            # Parse regular entries
                            try:
                                reader = csv.reader([line], delimiter="|")
                                row = next(reader)
                                # Support both old format (10 fields) and new format (11 fields with llm_code)
                                if len(row) == 10:
                                    # Old format: insert default llm_code "------" after msg_type
                                    row.insert(4, "------")
                                elif len(row) != 11:
                                    continue  # Skip malformed lines

                                entry = LogEntry(*row)
                                if entry.matches_filter(
                                    run_id=args.run_id,
                                    msg_type=args.type,
                                    agent=args.agent,
                                    llm=args.llm,
                                    status=args.status
                                ):
                                    if args.format == "json":
                                        print(format_entry_json(entry))
                                    else:
                                        print(format_entry_text(entry, color=not args.no_color))
                            except Exception:
                                continue  # Skip unparseable lines
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nStopped watching.", file=sys.stderr)
            sys.exit(0)

    # Parse log file
    entries = parse_log_file(log_path)

    if not entries:
        print(f"No entries found in {log_path}", file=sys.stderr)
        sys.exit(1)

    # Apply filters
    filtered = [
        e for e in entries
        if e.matches_filter(
            run_id=args.run_id,
            msg_type=args.type,
            agent=args.agent,
            llm=args.llm,
            status=args.status
        )
    ]

    # Apply limit
    if args.limit:
        filtered = filtered[-args.limit:]

    # Print results
    if args.format == "json":
        output = [e.to_dict() for e in filtered]
        print(json.dumps(output, indent=2))
    else:
        for entry in filtered:
            print(format_entry_text(entry, color=not args.no_color))

        # Print summary
        print(f"\n{len(filtered)} entries shown (out of {len(entries)} total)", file=sys.stderr)


if __name__ == "__main__":
    main()
