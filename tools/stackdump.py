#!/usr/bin/env python3
# tools/stackdump.py
"""
Usage:
  import tools.stackdump  # early in main
Then, from another shell:
  kill -USR1 <PID>
This will print all thread traces to STDERR without killing the process.
"""
import faulthandler, signal, sys

# Enable faulthandler globally and bind USR1 to dump all threads
faulthandler.enable(file=sys.stderr, all_threads=True)
try:
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
except Exception as e:
    # Some environments may not support SIGUSR1 (e.g., Windows). Best‑effort only.
    sys.stderr.write(f"[stackdump] SIGUSR1 registration failed: {e}\n")