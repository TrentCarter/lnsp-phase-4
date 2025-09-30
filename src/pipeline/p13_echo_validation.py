#!/usr/bin/env python3
"""
P13: Echo Validation Pipeline

Validates retrieval quality by comparing probe questions against retrieved concepts.
Tests that each CPE entry can be successfully retrieved using its probe question
with high cosine similarity (≥0.82).

This is NOT a cache validation - it's permanent training data validation for:
1. Ensuring CPESH quality before LVM training (P15)
2. Identifying low-quality lanes for re-interrogation
3. Generating quality scores for curriculum learning

Usage:
    python -m src.pipeline.p13_echo_validation \
        --batch-size 100 \
        --threshold 0.82 \
        --update-db \
        --report-out artifacts/p13_echo_report.json
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.db.pg_client import get_pg_connection
from src.vectorizer import EmbeddingBackend

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EchoValidationResult:
    """Result of echo validation for a single CPE entry."""
    cpe_id: str
    probe_question: str
    concept_text: str
    echo_score: float
    validation_status: str  # 'passed' | 'failed'
    domain_code: int
    task_code: int
    modifier_code: int
    tmd_lane: str


@dataclass
class ValidationReport:
    """Aggregate validation statistics."""
    total_entries: int
    validated_entries: int
    passed_entries: int
    failed_entries: int
    mean_echo_score: float
    median_echo_score: float
    p95_echo_score: float
    pass_rate: float
    failures_by_lane: Dict[str, int]
    scores_by_domain: Dict[int, float]
    low_quality_entries: List[Dict]


class P13EchoValidator:
    """Echo validation pipeline for P13."""

    def __init__(self, threshold: float = 0.82):
        """
        Initialize validator.

        Args:
            threshold: Minimum cosine similarity for passing (default 0.82)
        """
        self.threshold = threshold
        self.embedder = EmbeddingBackend()
        self.conn = get_pg_connection()
        logger.info(f"Initialized P13 validator with threshold={threshold}")

    def fetch_all_entries(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Fetch all CPE entries from database.

        Args:
            limit: Optional limit for testing

        Returns:
            List of CPE entries with necessary fields
        """
        query = """
            SELECT
                e.cpe_id,
                e.probe_question,
                e.concept_text,
                e.domain_code,
                e.task_code,
                e.modifier_code,
                e.tmd_lane,
                e.echo_score,
                e.validation_status
            FROM cpe_entry e
            WHERE e.probe_question IS NOT NULL
              AND e.probe_question != ''
            ORDER BY e.created_at
        """

        if limit:
            query += f" LIMIT {limit}"

        with self.conn.cursor() as cur:
            cur.execute(query)
            columns = [desc[0] for desc in cur.description]
            results = [dict(zip(columns, row)) for row in cur.fetchall()]

        logger.info(f"Fetched {len(results)} CPE entries for validation")
        return results

    def compute_echo_score(self, probe_question: str, concept_text: str) -> float:
        """
        Compute echo score (cosine similarity) between probe and concept.

        Args:
            probe_question: Probe question text
            concept_text: Concept text

        Returns:
            Cosine similarity score (0.0-1.0)
        """
        # Embed both texts
        probe_vec = self.embedder.encode([probe_question])[0]
        concept_vec = self.embedder.encode([concept_text])[0]

        # Compute cosine similarity
        dot_product = np.dot(probe_vec, concept_vec)
        norm_probe = np.linalg.norm(probe_vec)
        norm_concept = np.linalg.norm(concept_vec)

        if norm_probe == 0 or norm_concept == 0:
            return 0.0

        cosine_sim = dot_product / (norm_probe * norm_concept)
        return float(cosine_sim)

    def validate_entry(self, entry: Dict) -> EchoValidationResult:
        """
        Validate a single CPE entry.

        Args:
            entry: CPE entry dict from database

        Returns:
            EchoValidationResult with score and status
        """
        echo_score = self.compute_echo_score(
            entry['probe_question'],
            entry['concept_text']
        )

        validation_status = 'passed' if echo_score >= self.threshold else 'failed'

        return EchoValidationResult(
            cpe_id=str(entry['cpe_id']),
            probe_question=entry['probe_question'],
            concept_text=entry['concept_text'],
            echo_score=echo_score,
            validation_status=validation_status,
            domain_code=entry['domain_code'],
            task_code=entry['task_code'],
            modifier_code=entry['modifier_code'],
            tmd_lane=entry['tmd_lane']
        )

    def validate_batch(self, entries: List[Dict]) -> List[EchoValidationResult]:
        """
        Validate a batch of CPE entries.

        Args:
            entries: List of CPE entry dicts

        Returns:
            List of validation results
        """
        results = []
        for entry in tqdm(entries, desc="Validating", unit="entry"):
            try:
                result = self.validate_entry(entry)
                results.append(result)
            except Exception as e:
                logger.error(f"Error validating {entry['cpe_id']}: {e}")
                # Create failed result
                results.append(EchoValidationResult(
                    cpe_id=str(entry['cpe_id']),
                    probe_question=entry['probe_question'],
                    concept_text=entry['concept_text'],
                    echo_score=0.0,
                    validation_status='failed',
                    domain_code=entry['domain_code'],
                    task_code=entry['task_code'],
                    modifier_code=entry['modifier_code'],
                    tmd_lane=entry['tmd_lane']
                ))

        return results

    def update_database(self, results: List[EchoValidationResult]):
        """
        Update database with echo scores and validation status.

        Args:
            results: List of validation results
        """
        update_query = """
            UPDATE cpe_entry
            SET echo_score = %s,
                validation_status = %s
            WHERE cpe_id = %s
        """

        with self.conn.cursor() as cur:
            for result in tqdm(results, desc="Updating DB", unit="entry"):
                cur.execute(
                    update_query,
                    (result.echo_score, result.validation_status, result.cpe_id)
                )

        self.conn.commit()
        logger.info(f"Updated {len(results)} entries in database")

    def generate_report(self, results: List[EchoValidationResult]) -> ValidationReport:
        """
        Generate validation statistics report.

        Args:
            results: List of validation results

        Returns:
            ValidationReport with aggregate statistics
        """
        scores = [r.echo_score for r in results]
        passed = [r for r in results if r.validation_status == 'passed']
        failed = [r for r in results if r.validation_status == 'failed']

        # Failures by lane
        failures_by_lane = {}
        for result in failed:
            lane = result.tmd_lane
            failures_by_lane[lane] = failures_by_lane.get(lane, 0) + 1

        # Scores by domain
        scores_by_domain = {}
        domain_counts = {}
        for result in results:
            domain = result.domain_code
            if domain not in scores_by_domain:
                scores_by_domain[domain] = 0.0
                domain_counts[domain] = 0
            scores_by_domain[domain] += result.echo_score
            domain_counts[domain] += 1

        for domain in scores_by_domain:
            scores_by_domain[domain] /= domain_counts[domain]

        # Low quality entries (bottom 50 by score)
        sorted_results = sorted(results, key=lambda r: r.echo_score)
        low_quality = [
            {
                'cpe_id': r.cpe_id,
                'echo_score': r.echo_score,
                'tmd_lane': r.tmd_lane,
                'probe': r.probe_question[:100],
                'concept': r.concept_text[:100]
            }
            for r in sorted_results[:50]
        ]

        return ValidationReport(
            total_entries=len(results),
            validated_entries=len(results),
            passed_entries=len(passed),
            failed_entries=len(failed),
            mean_echo_score=float(np.mean(scores)),
            median_echo_score=float(np.median(scores)),
            p95_echo_score=float(np.percentile(scores, 95)),
            pass_rate=len(passed) / len(results) if results else 0.0,
            failures_by_lane=failures_by_lane,
            scores_by_domain={int(k): float(v) for k, v in scores_by_domain.items()},
            low_quality_entries=low_quality
        )

    def run(
        self,
        batch_size: int = 100,
        limit: Optional[int] = None,
        update_db: bool = True,
        report_path: Optional[Path] = None
    ) -> ValidationReport:
        """
        Run full P13 echo validation pipeline.

        Args:
            batch_size: Batch size for processing (not used currently)
            limit: Optional limit for testing
            update_db: Whether to update database with results
            report_path: Optional path to save JSON report

        Returns:
            ValidationReport with statistics
        """
        logger.info("=" * 80)
        logger.info("P13 ECHO VALIDATION PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Threshold: {self.threshold}")
        logger.info(f"Update DB: {update_db}")
        logger.info(f"Limit: {limit or 'None (all entries)'}")
        logger.info("")

        # Fetch entries
        entries = self.fetch_all_entries(limit=limit)

        if not entries:
            logger.error("No entries found to validate!")
            sys.exit(1)

        # Validate all entries
        results = self.validate_batch(entries)

        # Update database if requested
        if update_db:
            self.update_database(results)

        # Generate report
        report = self.generate_report(results)

        # Print summary
        self._print_summary(report)

        # Save report if path provided
        if report_path:
            self._save_report(report, report_path)

        return report

    def _print_summary(self, report: ValidationReport):
        """Print validation summary to console."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Entries:       {report.total_entries:,}")
        logger.info(f"Validated:           {report.validated_entries:,}")
        logger.info(f"Passed (≥{self.threshold}):    {report.passed_entries:,} ({report.pass_rate*100:.1f}%)")
        logger.info(f"Failed (<{self.threshold}):    {report.failed_entries:,} ({(1-report.pass_rate)*100:.1f}%)")
        logger.info("")
        logger.info(f"Mean Echo Score:     {report.mean_echo_score:.4f}")
        logger.info(f"Median Echo Score:   {report.median_echo_score:.4f}")
        logger.info(f"P95 Echo Score:      {report.p95_echo_score:.4f}")
        logger.info("")

        # Quality gate assessment
        if report.pass_rate >= 0.90:
            logger.info("✅ QUALITY GATE: PASSED (≥90% pass rate)")
        elif report.pass_rate >= 0.80:
            logger.warning("⚠️  QUALITY GATE: REVIEW NEEDED (80-90% pass rate)")
        else:
            logger.error("❌ QUALITY GATE: FAILED (<80% pass rate)")

        # Top failing lanes
        if report.failures_by_lane:
            logger.info("")
            logger.info("Top 10 Failing Lanes:")
            sorted_lanes = sorted(
                report.failures_by_lane.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            for lane, count in sorted_lanes:
                logger.info(f"  {lane}: {count} failures")

        # Scores by domain
        logger.info("")
        logger.info("Mean Scores by Domain:")
        for domain, score in sorted(report.scores_by_domain.items()):
            logger.info(f"  Domain {domain}: {score:.4f}")

        logger.info("=" * 80)

    def _save_report(self, report: ValidationReport, path: Path):
        """Save report to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)

        report_dict = asdict(report)

        with open(path, 'w') as f:
            json.dump(report_dict, f, indent=2)

        logger.info(f"Report saved to: {path}")

        # Also save failed entries JSONL
        failed_path = path.parent / f"{path.stem}_failed_entries.jsonl"
        with open(failed_path, 'w') as f:
            for entry in report.low_quality_entries:
                f.write(json.dumps(entry) + '\n')

        logger.info(f"Failed entries saved to: {failed_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="P13 Echo Validation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate all entries and update database
  python -m src.pipeline.p13_echo_validation --update-db --report-out artifacts/p13_report.json

  # Test run on 100 entries without updating
  python -m src.pipeline.p13_echo_validation --limit 100 --no-update-db

  # Custom threshold
  python -m src.pipeline.p13_echo_validation --threshold 0.85 --update-db
        """
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for processing (default: 100)'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.82,
        help='Cosine similarity threshold for passing (default: 0.82)'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of entries for testing (default: all)'
    )

    parser.add_argument(
        '--update-db',
        dest='update_db',
        action='store_true',
        help='Update database with echo scores (default: True)'
    )

    parser.add_argument(
        '--no-update-db',
        dest='update_db',
        action='store_false',
        help='Do not update database'
    )

    parser.add_argument(
        '--report-out',
        type=Path,
        default=None,
        help='Path to save JSON report (default: None)'
    )

    parser.set_defaults(update_db=True)

    args = parser.parse_args()

    # Run validation
    validator = P13EchoValidator(threshold=args.threshold)

    try:
        report = validator.run(
            batch_size=args.batch_size,
            limit=args.limit,
            update_db=args.update_db,
            report_path=args.report_out
        )

        # Exit code based on quality gate
        if report.pass_rate >= 0.90:
            sys.exit(0)  # Success
        elif report.pass_rate >= 0.80:
            sys.exit(1)  # Warning
        else:
            sys.exit(2)  # Failure

    except Exception as e:
        logger.exception(f"Fatal error during validation: {e}")
        sys.exit(3)


if __name__ == '__main__':
    main()
