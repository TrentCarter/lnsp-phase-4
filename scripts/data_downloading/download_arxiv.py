try:
    # Imported lazily when --extract-text is used and extractor=pdfminer
    from pdfminer.high_level import extract_text as pdf_extract_text  # type: ignore
except Exception:  # pragma: no cover
    pdf_extract_text = None

# Optional extractors
try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None

try:
    from bs4 import BeautifulSoup  # HTML parsing for ar5iv
except Exception:  # pragma: no cover
    BeautifulSoup = None
#!/usr/bin/env python3
"""
Download papers from arXiv API into JSONL under data/datasets/arxiv/.

- Uses the official arXiv API (Atom) at: https://export.arxiv.org/api/query
- Builds a search query from free-text and/or categories (e.g., cs.CL, cs.LG, stat.ML)
- Supports pagination until max_total results or API exhaustion
- Optional client-side date filtering (published range)
- Optional PDF downloads (saved under data/datasets/arxiv/pdfs/)

Example usages:
  python scripts/data_downloading/download_arxiv.py \
    --query "large language model" \
    --categories cs.CL,cs.LG \
    --max-total 500 --batch-size 100 \
    --out data/datasets/arxiv/llm_cs.jsonl

  python scripts/data_downloading/download_arxiv.py \
    --categories cs.CL \
    --from 2025-10-01 --until 2025-11-01 \
    --pdf --max-total 200

API reference (quick):
  - Endpoint: https://export.arxiv.org/api/query
  - Parameters:
      search_query: e.g., 'all:transformer AND (cat:cs.CL OR cat:cs.LG)'
      start: 0-based start index
      max_results: results per page (<= 2000 typically; recommended small, e.g., 100)
      sortBy: relevance|lastUpdatedDate|submittedDate
      sortOrder: ascending|descending

Output JSONL fields per paper:
  {
    "id": "http://arxiv.org/abs/1234.5678v1",
    "arxiv_id": "1234.5678v1",
    "title": str,
    "summary": str,
    "authors": ["Last, First", ...],
    "categories": ["cs.CL", "cs.LG"],
    "primary_category": "cs.CL",
    "published": "YYYY-MM-DDTHH:MM:SSZ",
    "updated": "YYYY-MM-DDTHH:MM:SSZ",
    "doi": str|None,
    "comment": str|None,
    "journal_ref": str|None,
    "links": {"abs": url, "pdf": url},
  }
"""

import argparse
import gzip
import datetime as dt
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def build_search_query(query: Optional[str], categories: Optional[List[str]]) -> str:
    parts = []
    if query:
        q = query.strip()
        if q:
            parts.append(f"all:{q}")
    if categories:
        cat_terms = [f"cat:{c.strip()}" for c in categories if c.strip()]
        if cat_terms:
            parts.append("(" + " OR ".join(cat_terms) + ")")
    if not parts:
        # arXiv requires a search_query; default to all:*
        return "all:*"
    return " AND ".join(parts)


def fetch_batch(params: Dict[str, str]) -> str:
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def parse_atom(xml_text: str) -> Dict[str, object]:
    root = ET.fromstring(xml_text)
    entries = []
    for entry in root.findall("atom:entry", ARXIV_NS):
        eid = entry.findtext("atom:id", default="", namespaces=ARXIV_NS)
        title = (entry.findtext("atom:title", default="", namespaces=ARXIV_NS) or "").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=ARXIV_NS) or "").strip()
        published = entry.findtext("atom:published", default="", namespaces=ARXIV_NS)
        updated = entry.findtext("atom:updated", default="", namespaces=ARXIV_NS)
        # Authors
        authors = []
        for a in entry.findall("atom:author", ARXIV_NS):
            name = a.findtext("atom:name", default="", namespaces=ARXIV_NS)
            if name:
                authors.append(name)
        # Links
        links = {"abs": None, "pdf": None}
        for l in entry.findall("atom:link", ARXIV_NS):
            href = l.attrib.get("href")
            rel = l.attrib.get("rel")
            title_attr = l.attrib.get("title")
            if href:
                if (rel == "alternate" and href.startswith("http")) or (title_attr == "doi"):
                    # arXiv "alternate" is usually the abstract/abs page
                    if "arxiv.org/abs/" in href:
                        links["abs"] = href
                if "arxiv.org/pdf/" in href:
                    links["pdf"] = href
        # arXiv extensions
        primary_category = None
        cat_elems = entry.findall("atom:category", ARXIV_NS)
        categories = []
        for c in cat_elems:
            term = c.attrib.get("term")
            if term:
                categories.append(term)
        pc_elem = entry.find("arxiv:primary_category", ARXIV_NS)
        if pc_elem is not None:
            primary_category = pc_elem.attrib.get("term")
        doi = None
        doi_elem = entry.find("arxiv:doi", ARXIV_NS)
        if doi_elem is not None and doi_elem.text:
            doi = doi_elem.text.strip()
        comment = None
        comment_elem = entry.find("arxiv:comment", ARXIV_NS)
        if comment_elem is not None and comment_elem.text:
            comment = comment_elem.text.strip()
        journal_ref = None
        j_elem = entry.find("arxiv:journal_ref", ARXIV_NS)
        if j_elem is not None and j_elem.text:
            journal_ref = j_elem.text.strip()
        # arxiv id
        arxiv_id = None
        if eid and "/abs/" in eid:
            arxiv_id = eid.split("/abs/")[-1]
        # build record
        rec = {
            "id": eid,
            "arxiv_id": arxiv_id,
            "title": title,
            "summary": summary,
            "authors": authors,
            "categories": categories,
            "primary_category": primary_category,
            "published": published,
            "updated": updated,
            "doi": doi,
            "comment": comment,
            "journal_ref": journal_ref,
            "links": links,
        }
        entries.append(rec)
    # totalResults (optional)
    total = None
    tr = root.find("{http://a9.com/-/spec/opensearch/1.1/}totalResults")
    if tr is not None and tr.text and tr.text.isdigit():
        total = int(tr.text)
    return {"entries": entries, "total": total}


def within_date_range(iso_ts: Optional[str], date_from: Optional[dt.date], date_until: Optional[dt.date]) -> bool:
    if not iso_ts or (not date_from and not date_until):
        return True
    try:
        # Example: 2025-10-31T17:05:04Z
        d = dt.datetime.fromisoformat(iso_ts.replace("Z", "+00:00")).date()
    except Exception:
        return True
    if date_from and d < date_from:
        return False
    if date_until and d > date_until:
        return False
    return True


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def iter_query(query: str, batch_size: int, max_total: int, sort_by: str, sort_order: str):
    start = 0
    fetched = 0
    total = None
    while True:
        params = {
            "search_query": query,
            "start": str(start),
            "max_results": str(batch_size),
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }
        xml = fetch_batch(params)
        parsed = parse_atom(xml)
        entries = parsed["entries"]
        total = parsed.get("total") if parsed.get("total") is not None else total
        if not entries:
            break
        for e in entries:
            yield e
            fetched += 1
            if fetched >= max_total:
                return
        start += batch_size
        # arXiv rate limiting: be gentle
        time.sleep(3.0)


def main():
    ap = argparse.ArgumentParser(description="Download arXiv metadata (and PDFs) into JSONL.")
    ap.add_argument("--query", type=str, default=None, help="Free-text search term for all:* (quotes recommended)")
    ap.add_argument("--categories", type=str, default=None, help="Comma-separated list of categories, e.g., cs.CL,cs.LG,stat.ML")
    ap.add_argument("--from", dest="date_from", type=str, default=None, help="Published from date (YYYY-MM-DD)")
    ap.add_argument("--until", dest="date_until", type=str, default=None, help="Published until date (YYYY-MM-DD)")
    ap.add_argument("--batch-size", type=int, default=100, help="Results per request (recommended <= 200)")
    ap.add_argument("--max-total", type=int, default=500, help="Maximum total records to fetch")
    ap.add_argument("--sort-by", type=str, default="submittedDate", choices=["relevance", "lastUpdatedDate", "submittedDate"])
    ap.add_argument("--sort-order", type=str, default="descending", choices=["ascending", "descending"])
    ap.add_argument("--out", type=str, default=None, help="Output JSONL path; default under data/datasets/arxiv/")
    ap.add_argument("--gzip", dest="use_gzip", action="store_true", help="Write compressed JSONL (.jsonl.gz)")
    ap.add_argument("--pdf", action="store_true", help="Also download PDFs under data/datasets/arxiv/pdfs/")
    ap.add_argument("--extract-text", action="store_true", help="Extract text into .txt and add 'fulltext_path' to JSONL")
    ap.add_argument("--extractor", type=str, default="pdfminer", choices=["pdfminer","pymupdf","ar5iv"], help="Extraction backend: pdfminer (default), pymupdf, or ar5iv (HTML)")
    ap.add_argument("--inline-text", action="store_true", help="When used with --extract-text, also include 'fulltext' in each JSONL record")
    args = ap.parse_args()

    cats = [c.strip() for c in (args.categories.split(",") if args.categories else []) if c.strip()]
    search_query = build_search_query(args.query, cats)

    date_from = dt.date.fromisoformat(args.date_from) if args.date_from else None
    date_until = dt.date.fromisoformat(args.date_until) if args.date_until else None

    # Output paths
    base_dir = os.path.join("data", "datasets", "arxiv")
    ensure_dir(base_dir)
    if args.out:
        out_path = args.out
        ensure_dir(os.path.dirname(out_path))
    else:
        name_bits = []
        if args.query:
            name_bits.append(args.query.replace(" ", "_")[:32])
        if cats:
            name_bits.append("_".join(cats))
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = "_".join([b for b in name_bits if b]) or "arxiv"
        ext = "jsonl.gz" if args.use_gzip else "jsonl"
        out_path = os.path.join(base_dir, f"{base_name}_{ts}.{ext}")
    print(f"Writing to: {out_path}")

    pdf_dir = os.path.join(base_dir, "pdfs") if args.pdf else None
    if pdf_dir:
        ensure_dir(pdf_dir)
    if args.extract_text:
        if args.extractor in ("pdfminer","pymupdf") and not args.pdf:
            print("[WARN] --extract-text with PDF extractor but --pdf not set; enabling --pdf automatically.")
            pdf_dir = os.path.join(base_dir, "pdfs")
            ensure_dir(pdf_dir)
            args.pdf = True
        if args.extractor == "pdfminer" and pdf_extract_text is None:
            print("[ERROR] pdfminer.six not installed. pip install pdfminer.six", file=sys.stderr)
            sys.exit(1)
        if args.extractor == "pymupdf" and fitz is None:
            print("[ERROR] PyMuPDF not installed. pip install pymupdf", file=sys.stderr)
            sys.exit(1)
        if args.extractor == "ar5iv" and BeautifulSoup is None:
            print("[ERROR] bs4 not installed. pip install beautifulsoup4", file=sys.stderr)
            sys.exit(1)

    def _normalize_text(txt: str) -> str:
        if not txt:
            return ""
        # Collapse excessive newlines/spaces
        txt = txt.replace('\r', '\n')
        # Heuristic: merge lines consisting of single characters
        lines = [ln.strip() for ln in txt.split('\n')]
        if lines and sum(1 for ln in lines if len(ln) <= 2) > 50:
            txt = " ".join([ln for ln in lines if ln])
        else:
            txt = "\n".join(lines)
        # Collapse repeated whitespace
        import re
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt

    def _extract_text_pdfminer(pdf_path: str) -> str:
        return _normalize_text(pdf_extract_text(pdf_path) if pdf_extract_text else "")

    def _extract_text_pymupdf(pdf_path: str) -> str:
        if fitz is None:
            return ""
        text_parts = []
        doc = fitz.open(pdf_path)
        for page in doc:
            text_parts.append(page.get_text("text"))
        doc.close()
        return _normalize_text("\n".join(text_parts))

    def _extract_text_ar5iv(arxiv_id: str) -> str:
        # Fetch HTML from ar5iv and extract text
        url = f"https://ar5iv.org/html/{arxiv_id}"
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
            if BeautifulSoup is None:
                return ""
            soup = BeautifulSoup(html, "html.parser")
            # Remove nav/figures/scripts
            for tag in soup(["script","style","nav","header","footer"]):
                tag.decompose()
            return _normalize_text(soup.get_text(separator="\n"))
        except Exception:
            return ""

    # Iterate and write
    count = 0
    open_fn = (lambda p: gzip.open(p, "wt", encoding="utf-8")) if args.use_gzip else (lambda p: open(p, "w", encoding="utf-8"))
    with open_fn(out_path) as f:
        for rec in iter_query(search_query, args.batch_size, args.max_total, args.sort_by, args.sort_order):
            # Client-side date filter on published
            if not within_date_range(rec.get("published"), date_from, date_until):
                continue
            # Download PDF / extract text first so JSONL contains these fields
            if (args.extract_text and args.extractor == "ar5iv") or (pdf_dir and rec.get("links", {}).get("pdf")):
                try:
                    arxiv_id = (rec.get("arxiv_id") or f"paper_{count}").replace("/", "_")
                    txt_path = None
                    text = ""
                    if args.extract_text and args.extractor == "ar5iv":
                        text = _extract_text_ar5iv(arxiv_id)
                        if text:
                            # Store as .txt under pdf_dir for consistency
                            if not pdf_dir:
                                pdf_dir = os.path.join(base_dir, "pdfs")
                                ensure_dir(pdf_dir)
                            txt_path = os.path.join(pdf_dir, f"{arxiv_id}.txt")
                            with open(txt_path, "w", encoding="utf-8") as tf:
                                tf.write(text)
                    else:
                        # PDF path + extractor (pdfminer/pymupdf)
                        if pdf_dir and rec.get("links", {}).get("pdf"):
                            pdf_url = rec["links"]["pdf"]
                            if not pdf_url.lower().endswith(".pdf") and "/pdf/" in pdf_url:
                                pdf_url = pdf_url if pdf_url.endswith(".pdf") else pdf_url + ".pdf"
                            pdf_path = os.path.join(pdf_dir, f"{arxiv_id}.pdf")
                            if not os.path.exists(pdf_path):
                                urllib.request.urlretrieve(pdf_url, pdf_path)
                                time.sleep(1.0)  # be nice
                            if args.extract_text:
                                if args.extractor == "pdfminer":
                                    text = _extract_text_pdfminer(pdf_path)
                                elif args.extractor == "pymupdf":
                                    text = _extract_text_pymupdf(pdf_path)
                                txt_path = os.path.join(pdf_dir, f"{arxiv_id}.txt")
                                with open(txt_path, "w", encoding="utf-8") as tf:
                                    tf.write(text or "")

                    # Fallback: if text too short/degenerate, try ar5iv
                    if args.extract_text and (not text or len(text) < 500):
                        alt = _extract_text_ar5iv(arxiv_id)
                        if alt and len(alt) > len(text):
                            text = alt
                            if not pdf_dir:
                                pdf_dir = os.path.join(base_dir, "pdfs")
                                ensure_dir(pdf_dir)
                            txt_path = os.path.join(pdf_dir, f"{arxiv_id}.txt")
                            with open(txt_path, "w", encoding="utf-8") as tf:
                                tf.write(text)

                    if args.extract_text:
                        rec["fulltext_path"] = txt_path
                        if args.inline_text and text:
                            rec["fulltext"] = text
                except Exception as e:
                    print(f"PDF download failed for {rec.get('arxiv_id')}: {e}", file=sys.stderr)
            # Now write the (possibly augmented) record
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} records to {out_path}")


if __name__ == "__main__":
    main()
