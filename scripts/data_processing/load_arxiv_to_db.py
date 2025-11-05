#!/usr/bin/env python3
"""
Load arXiv JSONL(.gz) into Postgres table `arxiv_papers`.

- Accepts a path to .jsonl or .jsonl.gz.
- Creates table if missing.
- Inserts rows with metadata and optional fulltext/fulltext_path.

Env: PG_DSN (defaults to 'host=localhost dbname=lnsp user=lnsp password=lnsp')

Usage:
  python scripts/data_processing/load_arxiv_to_db.py \
    --input data/datasets/arxiv/arxiv_cs_lg_ml.jsonl.gz

Optionally select only records that have extracted text:
  python scripts/data_processing/load_arxiv_to_db.py \
    --input data/datasets/arxiv/arxiv_cs_lg_ml.jsonl.gz --require-text
"""

import argparse
import gzip
import json
import os
import sys
from typing import Iterator, Dict, Any

PG_DSN = os.getenv("PG_DSN", "host=localhost dbname=lnsp user=lnsp password=lnsp")

try:
    import psycopg2
    import psycopg2.extras
except Exception:
    psycopg2 = None


def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    open_fn = gzip.open if path.endswith(".gz") else open
    mode = "rt" if path.endswith(".gz") else "r"
    with open_fn(path, mode, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                print(f"[WARN] Skip line parse error: {e}", file=sys.stderr)


def ensure_table(conn):
    sql = """
    CREATE TABLE IF NOT EXISTS arxiv_papers (
        id TEXT PRIMARY KEY,
        arxiv_id TEXT,
        title TEXT,
        summary TEXT,
        authors JSONB,
        categories JSONB,
        primary_category TEXT,
        published TIMESTAMPTZ NULL,
        updated TIMESTAMPTZ NULL,
        doi TEXT NULL,
        comment TEXT NULL,
        journal_ref TEXT NULL,
        abs_url TEXT NULL,
        pdf_url TEXT NULL,
        fulltext_path TEXT NULL,
        fulltext TEXT NULL
    );
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        conn.commit()


def insert_row(conn, rec: Dict[str, Any]):
    sql = """
    INSERT INTO arxiv_papers(
        id, arxiv_id, title, summary, authors, categories, primary_category,
        published, updated, doi, comment, journal_ref, abs_url, pdf_url, fulltext_path, fulltext
    ) VALUES (
        %(id)s, %(arxiv_id)s, %(title)s, %(summary)s, %(authors)s, %(categories)s, %(primary_category)s,
        %(published)s, %(updated)s, %(doi)s, %(comment)s, %(journal_ref)s, %(abs_url)s, %(pdf_url)s, %(fulltext_path)s, %(fulltext)s
    ) ON CONFLICT (id) DO NOTHING;
    """
    # map fields
    payload = {
        "id": rec.get("id"),
        "arxiv_id": rec.get("arxiv_id"),
        "title": rec.get("title"),
        "summary": rec.get("summary"),
        "authors": json.dumps(rec.get("authors", [])),
        "categories": json.dumps(rec.get("categories", [])),
        "primary_category": rec.get("primary_category"),
        "published": rec.get("published"),
        "updated": rec.get("updated"),
        "doi": rec.get("doi"),
        "comment": rec.get("comment"),
        "journal_ref": rec.get("journal_ref"),
        "abs_url": (rec.get("links") or {}).get("abs"),
        "pdf_url": (rec.get("links") or {}).get("pdf"),
        "fulltext_path": rec.get("fulltext_path"),
        "fulltext": rec.get("fulltext"),
    }
    with conn.cursor() as cur:
        cur.execute(sql, payload)


def main():
    ap = argparse.ArgumentParser(description="Load arXiv JSONL(.gz) into Postgres arxiv_papers table")
    ap.add_argument("--input", required=True, help="Path to .jsonl or .jsonl.gz")
    ap.add_argument("--require-text", action="store_true", help="Only load records with fulltext_path or fulltext")
    args = ap.parse_args()

    if psycopg2 is None:
        print("[ERROR] psycopg2 not installed. pip install psycopg2-binary", file=sys.stderr)
        sys.exit(1)

    conn = psycopg2.connect(PG_DSN)
    conn.autocommit = True
    ensure_table(conn)

    total = 0
    loaded = 0
    for rec in read_jsonl(args.input):
        total += 1
        if args.require-text and not (rec.get("fulltext") or rec.get("fulltext_path")):
            continue
        insert_row(conn, rec)
        loaded += 1
        if loaded % 100 == 0:
            print(f"Loaded {loaded} / seen {total}")
    conn.commit()
    conn.close()
    print(f"Done. Loaded {loaded} rows (seen {total}).")


if __name__ == "__main__":
    main()
