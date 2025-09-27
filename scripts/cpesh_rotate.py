#!/usr/bin/env python
import os, time, json, shutil
import pyarrow as pa, pyarrow.json as paj, pyarrow.parquet as pq

ACTIVE = "artifacts/cpesh_active.jsonl"   # or cpesh_cache.jsonl while you transition
SEGDIR = "artifacts/cpesh_segments"
MANIFEST = "artifacts/cpesh_manifest.jsonl"
MAX_LINES = int(os.getenv("CPESH_ROTATE_MAX_LINES", "1000000"))
MAX_BYTES = int(os.getenv("CPESH_ROTATE_MAX_BYTES", "104857600"))

def approx_lines(path, hard=False):
    if not os.path.exists(path): return 0
    if hard:
        return sum(1 for _ in open(path, "r", encoding="utf-8"))
    # quick heuristic by size/avg line len (fallback to hard if small)
    sz = os.path.getsize(path)
    return 0 if sz == 0 else max(1, sz // 256)

def should_rotate():
    if not os.path.exists(ACTIVE): return False
    return (os.path.getsize(ACTIVE) >= MAX_BYTES) or (approx_lines(ACTIVE, hard=True) >= MAX_LINES)

def rotate():
    os.makedirs(SEGDIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    seg_id = f"cpesh_{ts}"
    tmp_json = os.path.join(SEGDIR, seg_id + ".jsonl")
    tmp_parq = os.path.join(SEGDIR, seg_id + ".parquet")

    shutil.move(ACTIVE, tmp_json)
    # ingest JSONL -> Arrow -> Parquet (columnar, compressed)
    # Use explicit_schema=False to handle mixed types
    try:
        table = paj.read_json(tmp_json)
    except Exception:
        # Fallback: read line by line and convert
        import pandas as pd
        with open(tmp_json, 'r') as f:
            data = [json.loads(line) for line in f]
        df = pd.DataFrame(data)
        table = pa.Table.from_pandas(df)
    pq.write_table(table, tmp_parq, compression="zstd", use_dictionary=True)
    os.remove(tmp_json)

    meta = {
        "segment_id": seg_id,
        "path": tmp_parq,
        "rows": table.num_rows,
        "cols": table.num_columns,
        "created_utc": ts,
    }
    with open(MANIFEST, "a") as mf:
        mf.write(json.dumps(meta) + "\n")

    # recreate empty ACTIVE
    open(ACTIVE, "a").close()
    print(f"[cpesh-rotate] wrote {tmp_parq} rows={meta['rows']}")

if __name__ == "__main__":
    if should_rotate():
        rotate()
    else:
        print("[cpesh-rotate] no rotation needed")
