#!/usr/bin/env python3
"""
Monitor vecRAG data ingestion integrity every 100 articles.

This script checks for data loss by:
1. Counting total articles and chunks in database
2. Identifying gaps in article indices (missing articles)
3. Checking chunk counts per article for consistency
4. Sampling articles to verify content integrity
5. Logging results for tracking

Usage:
    # Check current status
    python tools/monitor_ingestion.py --status

    # Check every 100 articles
    python tools/monitor_ingestion.py --every 100 --output monitor_log.json

    # Specific article range check
    python tools/monitor_ingestion.py --start 1 --end 500 --output range_check.json
"""

import argparse
import json
import psycopg2
import psycopg2.extras
import sys
from typing import Dict, List, Set
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.db_postgres import connect as connect_pg


def get_article_stats(conn) -> Dict:
    """Get comprehensive article statistics."""
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # Total counts
    cur.execute("""
        SELECT
            COUNT(DISTINCT (chunk_position->>'source')) as total_articles,
            COUNT(*) as total_chunks,
            COUNT(DISTINCT chunk_position->>'article_index') as unique_article_indices,
            MIN((chunk_position->>'article_index')::int) as min_article_index,
            MAX((chunk_position->>'article_index')::int) as max_article_index
        FROM cpe_entry
        WHERE dataset_source = 'wikipedia_500k'
    """)
    stats = cur.fetchone()
    cur.close()

    return dict(stats)


def find_missing_articles(conn, min_idx: int, max_idx: int) -> List[int]:
    """Find missing article indices in the range."""
    cur = conn.cursor()

    query = """
        SELECT DISTINCT (chunk_position->>'article_index')::int as article_idx
        FROM cpe_entry
        WHERE dataset_source = 'wikipedia_500k'
          AND (chunk_position->>'article_index')::int BETWEEN %s AND %s
        ORDER BY article_idx
    """

    cur.execute(query, (min_idx, max_idx))
    results = cur.fetchall()
    cur.close()

    # Find gaps
    found_indices = {row[0] for row in results}
    missing_indices = []

    for idx in range(min_idx, max_idx + 1):
        if idx not in found_indices:
            missing_indices.append(idx)

    return missing_indices


def get_chunk_counts_per_article(conn, article_indices: List[int]) -> Dict[int, int]:
    """Get chunk count for each article."""
    if not article_indices:
        return {}

    cur = conn.cursor()

    # Build query with IN clause
    placeholders = ','.join(['%s'] * len(article_indices))
    query = f"""
        SELECT
            (chunk_position->>'article_index')::int as article_idx,
            COUNT(*) as chunk_count
        FROM cpe_entry
        WHERE dataset_source = 'wikipedia_500k'
          AND (chunk_position->>'article_index')::int IN ({placeholders})
        GROUP BY (chunk_position->>'article_index')::int
        ORDER BY article_idx
    """

    cur.execute(query, article_indices)
    results = cur.fetchall()
    cur.close()

    return {row[0]: row[1] for row in results}


def sample_article_content(conn, article_index: int, max_chunks: int = 5) -> Dict:
    """Sample a few chunks from an article for content verification."""
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    query = """
        SELECT
            cpe_id,
            concept_text,
            chunk_position,
            created_at
        FROM cpe_entry
        WHERE dataset_source = 'wikipedia_500k'
          AND chunk_position->>'article_index' = %s
        ORDER BY (chunk_position->>'chunk_index')::int
        LIMIT %s
    """

    cur.execute(query, (str(article_index), max_chunks))
    results = cur.fetchall()
    cur.close()

    return [dict(row) for row in results]


def check_data_integrity(conn, start_idx: int = None, end_idx: int = None) -> Dict:
    """Comprehensive data integrity check."""
    print(f"🔍 Checking data integrity...")

    # Get overall stats
    stats = get_article_stats(conn)
    print(f"📊 Found {stats['total_articles']} articles, {stats['total_chunks']} chunks")
    print(f"📈 Article range: {stats['min_article_index']} to {stats['max_article_index']}")

    # Determine range to check
    if start_idx is None:
        start_idx = stats['min_article_index']
    if end_idx is None:
        end_idx = stats['max_article_index']

    print(f"🔎 Checking range: articles {start_idx} to {end_idx}")

    # Find missing articles
    missing_articles = find_missing_articles(conn, start_idx, end_idx)
    print(f"⚠️  Missing articles: {len(missing_articles)} gaps found")

    if missing_articles:
        print(f"   Missing: {missing_articles[:10]}{'...' if len(missing_articles) > 10 else ''}")

    # Check chunk counts for existing articles
    all_indices_in_range = list(range(start_idx, end_idx + 1))
    chunk_counts = get_chunk_counts_per_article(conn, all_indices_in_range)

    # Analyze chunk distribution
    if chunk_counts:
        counts = list(chunk_counts.values())
        min_chunks = min(counts)
        max_chunks = max(counts)
        avg_chunks = sum(counts) / len(counts)

        print(f"📋 Chunk distribution: min={min_chunks}, max={max_chunks}, avg={avg_chunks:.1f}")

        # Find articles with unusually low/high chunk counts
        low_chunk_articles = [(idx, count) for idx, count in chunk_counts.items() if count < avg_chunks * 0.5]
        high_chunk_articles = [(idx, count) for idx, count in chunk_counts.items() if count > avg_chunks * 2]

        if low_chunk_articles:
            print(f"⚠️  Low chunks: {low_chunk_articles[:5]}")

        if high_chunk_articles:
            print(f"⚠️  High chunks: {high_chunk_articles[:5]}")

    # Sample a few articles for content verification
    print(f"\n🧪 Sampling articles for content verification...")
    sample_indices = []
    if chunk_counts:
        # Sample from beginning, middle, and end of range
        sample_indices = [
            start_idx,
            start_idx + (end_idx - start_idx) // 2,
            end_idx
        ]
        # Add a few random samples
        import random
        available_indices = [idx for idx in chunk_counts.keys() if start_idx <= idx <= end_idx]
        if len(available_indices) > 3:
            sample_indices.extend(random.sample(available_indices, min(3, len(available_indices) - 3)))

    samples = {}
    for idx in sample_indices:
        if idx in chunk_counts:
            sample = sample_article_content(conn, idx)
            # Convert datetime objects to strings for JSON serialization
            for chunk in sample:
                if 'created_at' in chunk and chunk['created_at']:
                    chunk['created_at'] = chunk['created_at'].isoformat()
            samples[idx] = {
                'chunk_count': chunk_counts[idx],
                'sample_chunks': sample
            }

    # Compile results
    integrity_report = {
        'timestamp': datetime.now().isoformat(),
        'range_checked': {'start': start_idx, 'end': end_idx},
        'stats': stats,
        'missing_articles': missing_articles,
        'chunk_counts': chunk_counts,
        'samples': samples,
        'issues': {
            'missing_count': len(missing_articles),
            'low_chunk_articles': low_chunk_articles,
            'high_chunk_articles': high_chunk_articles
        }
    }

    return integrity_report


def main():
    parser = argparse.ArgumentParser(description="Monitor vecRAG ingestion integrity")
    parser.add_argument("--status", action="store_true", help="Quick status check")
    parser.add_argument("--every", type=int, help="Check every N articles (e.g., 100)")
    parser.add_argument("--start", type=int, help="Start article index")
    parser.add_argument("--end", type=int, help="End article index")
    parser.add_argument("--output", help="Output JSON file for detailed report")
    parser.add_argument("--sample-size", type=int, default=5, help="Number of articles to sample")

    args = parser.parse_args()

    print("🔍 vecRAG Data Integrity Monitor")
    print("=" * 50)

    # Connect to database
    print("Connecting to PostgreSQL...")
    try:
        conn = connect_pg()
        print("✅ Connected successfully")
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        return 1

    try:
        if args.status:
            # Quick status check
            stats = get_article_stats(conn)
            print("📊 Current Status:")
            print(f"   Articles: {stats['total_articles']}")
            print(f"   Chunks: {stats['total_chunks']}")
            print(f"   Range: {stats['min_article_index']} to {stats['max_article_index']}")

            # Quick missing check for current range
            missing = find_missing_articles(conn, stats['min_article_index'], stats['max_article_index'])
            print(f"   Missing: {len(missing)} articles")

        else:
            # Comprehensive integrity check
            report = check_data_integrity(conn, args.start, args.end)

            # Print summary
            print(f"\n📋 Integrity Check Summary:")
            print(f"   Time: {report['timestamp']}")
            print(f"   Range: {report['range_checked']['start']} to {report['range_checked']['end']}")
            print(f"   Articles found: {report['stats']['total_articles']}")
            print(f"   Chunks found: {report['stats']['total_chunks']}")
            print(f"   Missing articles: {report['issues']['missing_count']}")

            if report['issues']['missing_count'] > 0:
                print(f"   ⚠️  DATA LOSS DETECTED!")
                print(f"   Missing: {len(report['missing_articles'])} articles")
                if report['missing_articles']:
                    print(f"   Examples: {report['missing_articles'][:10]}")

            # Save detailed report if requested
            if args.output:
                print(f"\n💾 Saving detailed report to {args.output}...")
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                print("✅ Report saved")

            # Continuous monitoring if --every specified
            if args.every:
                print(f"\n🔄 Continuous monitoring every {args.every} articles...")
                print("   Press Ctrl+C to stop")
                import time

                last_max = report['stats']['max_article_index']
                while True:
                    time.sleep(30)  # Check every 30 seconds

                    current_report = check_data_integrity(conn, args.start, args.end)
                    current_max = current_report['stats']['max_article_index']

                    if current_max > last_max:
                        print(f"\n📈 Progress: {last_max} → {current_max} articles")
                        last_max = current_max

                        # Check if we've hit the next 100 milestone
                        if current_max // args.every > last_max // args.every:
                            print(f"🎯 Hit {args.every} article milestone! Running integrity check...")

                            # Run detailed check at this milestone
                            milestone_report = check_data_integrity(conn, args.start, args.end)
                            if args.output:
                                milestone_file = args.output.replace('.json', f'_{current_max}.json')
                                with open(milestone_file, 'w', encoding='utf-8') as f:
                                    json.dump(milestone_report, f, indent=2)
                                print(f"💾 Milestone report saved: {milestone_file}")

                    # Check for data loss
                    current_missing = len(current_report['missing_articles'])
                    if current_missing > len(report['missing_articles']):
                        print(f"⚠️  NEW DATA LOSS DETECTED! Missing articles increased to {current_missing}")

    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped by user")
    except Exception as e:
        print(f"❌ Error during monitoring: {e}")
        return 1
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
