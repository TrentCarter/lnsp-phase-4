import os, psutil, time

def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

def log_mem(prefix="mem"):
    print(f"[{prefix}] rss_mb={rss_mb():.1f}")
