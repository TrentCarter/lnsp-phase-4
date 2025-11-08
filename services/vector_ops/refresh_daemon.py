#!/usr/bin/env python3
"""
Vector-Ops Refresh Daemon

Monitors git events and automatically triggers LightRAG index refresh
when code changes are detected. Enforces freshness SLO (≤2 minutes from commit).

Usage:
    python services/vector_ops/refresh_daemon.py --repo . --index-dir artifacts/lightrag_code_index
"""
import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IndexFreshnessMonitor:
    """Monitors LightRAG index freshness and triggers refresh when stale."""

    def __init__(
        self,
        repo_path: Path,
        index_dir: Path,
        freshness_slo_seconds: int = 120,
        check_interval_seconds: int = 30
    ):
        self.repo_path = repo_path
        self.index_dir = index_dir
        self.freshness_slo = timedelta(seconds=freshness_slo_seconds)
        self.check_interval = check_interval_seconds
        self.last_refresh_time: Optional[datetime] = None
        self.last_commit_sha: Optional[str] = None

    def get_latest_commit(self) -> tuple[str, datetime]:
        """Get latest commit SHA and timestamp from git."""
        try:
            # Get latest commit SHA
            result = subprocess.run(
                ["git", "-C", str(self.repo_path), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True
            )
            sha = result.stdout.strip()

            # Get commit timestamp
            result = subprocess.run(
                ["git", "-C", str(self.repo_path), "show", "-s", "--format=%ct", sha],
                capture_output=True,
                text=True,
                check=True
            )
            timestamp = int(result.stdout.strip())
            commit_time = datetime.fromtimestamp(timestamp)

            return sha, commit_time

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get latest commit: {e}")
            raise

    def get_index_timestamp(self) -> Optional[datetime]:
        """Get timestamp of last index update from metadata file."""
        metadata_file = self.index_dir / "metadata.json"

        if not metadata_file.exists():
            logger.warning(f"Index metadata not found: {metadata_file}")
            return None

        try:
            with open(metadata_file) as f:
                metadata = json.load(f)
                timestamp_str = metadata.get("last_update")
                if timestamp_str:
                    return datetime.fromisoformat(timestamp_str)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to read index metadata: {e}")

        return None

    def is_stale(self) -> tuple[bool, timedelta]:
        """
        Check if index is stale (older than freshness SLO).

        Returns:
            (is_stale, age) tuple
        """
        commit_sha, commit_time = self.get_latest_commit()
        index_time = self.get_index_timestamp()

        if index_time is None:
            # No index exists, definitely stale
            return True, timedelta(days=999)

        if commit_sha == self.last_commit_sha and self.last_refresh_time:
            # Same commit, use last refresh time
            age = datetime.now() - self.last_refresh_time
        else:
            # New commit, calculate age from commit time
            age = datetime.now() - commit_time

        is_stale = age > self.freshness_slo

        if is_stale:
            logger.warning(
                f"Index STALE: age={age.total_seconds():.1f}s > SLO={self.freshness_slo.total_seconds():.0f}s"
            )
        else:
            logger.debug(
                f"Index FRESH: age={age.total_seconds():.1f}s ≤ SLO={self.freshness_slo.total_seconds():.0f}s"
            )

        return is_stale, age

    def refresh_index(self) -> bool:
        """
        Trigger LightRAG index refresh.

        Returns:
            True if refresh succeeded, False otherwise
        """
        commit_sha, commit_time = self.get_latest_commit()

        logger.info(f"Starting index refresh for commit {commit_sha[:8]}...")
        start_time = time.time()

        try:
            # TODO: Replace with actual LightRAG refresh command
            # For now, this is a stub that creates metadata
            result = subprocess.run(
                [
                    "python", "-c",
                    f"""
import json
from pathlib import Path
from datetime import datetime

index_dir = Path("{self.index_dir}")
index_dir.mkdir(parents=True, exist_ok=True)

metadata = {{
    "last_update": datetime.now().isoformat(),
    "commit_sha": "{commit_sha}",
    "commit_time": "{commit_time.isoformat()}",
    "refresh_duration_seconds": 0.0
}}

with open(index_dir / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Index refreshed: {{index_dir}}")
"""
                ],
                capture_output=True,
                text=True,
                timeout=300  # 5 min timeout
            )

            if result.returncode != 0:
                logger.error(f"Index refresh failed: {result.stderr}")
                return False

            duration = time.time() - start_time
            logger.info(f"✅ Index refresh completed in {duration:.1f}s")

            # Update tracking state
            self.last_refresh_time = datetime.now()
            self.last_commit_sha = commit_sha

            # Update metadata with actual duration
            metadata_file = self.index_dir / "metadata.json"
            with open(metadata_file) as f:
                metadata = json.load(f)
            metadata["refresh_duration_seconds"] = duration
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            return True

        except subprocess.TimeoutExpired:
            logger.error("Index refresh timed out (300s)")
            return False
        except Exception as e:
            logger.error(f"Index refresh failed: {e}")
            return False

    def run(self):
        """Main daemon loop - check freshness and refresh if needed."""
        logger.info(f"Starting Vector-Ops Refresh Daemon")
        logger.info(f"  Repo: {self.repo_path}")
        logger.info(f"  Index: {self.index_dir}")
        logger.info(f"  Freshness SLO: {self.freshness_slo.total_seconds():.0f}s")
        logger.info(f"  Check Interval: {self.check_interval}s")

        try:
            while True:
                try:
                    is_stale, age = self.is_stale()

                    if is_stale:
                        logger.warning(f"Index stale (age={age.total_seconds():.1f}s) - triggering refresh")
                        success = self.refresh_index()
                        if not success:
                            logger.error("Refresh failed - will retry on next check")
                    else:
                        logger.info(f"Index fresh (age={age.total_seconds():.1f}s)")

                except Exception as e:
                    logger.error(f"Error during freshness check: {e}")

                # Sleep until next check
                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            logger.info("Shutting down daemon (Ctrl+C)")
            sys.exit(0)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Vector-Ops Refresh Daemon - Monitor and refresh LightRAG index"
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path("."),
        help="Git repository path (default: .)"
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("artifacts/lightrag_code_index"),
        help="LightRAG index directory (default: artifacts/lightrag_code_index)"
    )
    parser.add_argument(
        "--freshness-slo",
        type=int,
        default=120,
        help="Freshness SLO in seconds (default: 120)"
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=30,
        help="Check interval in seconds (default: 30)"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit (don't daemonize)"
    )

    args = parser.parse_args()

    monitor = IndexFreshnessMonitor(
        repo_path=args.repo,
        index_dir=args.index_dir,
        freshness_slo_seconds=args.freshness_slo,
        check_interval_seconds=args.check_interval
    )

    if args.once:
        # Run once and exit
        is_stale, age = monitor.is_stale()
        if is_stale:
            logger.info(f"Index stale (age={age.total_seconds():.1f}s) - refreshing")
            success = monitor.refresh_index()
            sys.exit(0 if success else 1)
        else:
            logger.info(f"Index fresh (age={age.total_seconds():.1f}s)")
            sys.exit(0)
    else:
        # Run as daemon
        monitor.run()


if __name__ == "__main__":
    main()
