#!/usr/bin/env python3
"""Quick fix: Rule-based TMD assignment for ontology sources."""
import sys
from pathlib import Path
import psycopg2
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.tmd_encoder import pack_tmd, tmd16_deterministic

# Source-based TMD mapping
TMD_MAP = {
    'swo': (2, 1, 0),      # engineering, code_generation, neutral
    'go': (4, 0, 0),       # medicine, fact_retrieval, neutral
    'dbpedia': (9, 0, 0),  # art, fact_retrieval, neutral
    None: (0, 0, 0)        # default: science, fact_retrieval, neutral
}

print('🔧 Fixing TMD vectors for ontology concepts...')

conn = psycopg2.connect('dbname=lnsp')
cur = conn.cursor()

# Get all concepts
cur.execute("""
    SELECT v.cpe_id, e.dataset_source
    FROM cpe_vectors v
    JOIN cpe_entry e ON v.cpe_id = e.cpe_id
    WHERE v.tmd_dense IS NOT NULL
""")

rows = cur.fetchall()
print(f'Processing {len(rows)} concepts...')

for i, (cpe_id, dataset_source) in enumerate(rows):
    if i % 500 == 0:
        print(f'Progress: {i}/{len(rows)} ({100*i/len(rows):.1f}%)')

    # Extract source from dataset_source (format: "swo_chains", "go_chains", etc.)
    source = dataset_source.split('_')[0] if dataset_source else None
    domain, task, modifier = TMD_MAP.get(source, TMD_MAP[None])

    # Create 16D dense vector
    tmd_dense = tmd16_deterministic(domain, task, modifier)

    cur.execute("""
        UPDATE cpe_vectors
        SET tmd_dense = %s
        WHERE cpe_id = %s
    """, (json.dumps(tmd_dense.tolist()), cpe_id))

conn.commit()
print('✅ All TMD vectors updated!')

# Verify
cur.execute("""
    SELECT COUNT(*)
    FROM cpe_vectors
    WHERE tmd_dense::jsonb = '[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'::jsonb
""")
zero_count = cur.fetchone()[0]
print(f'\n📊 Zero TMD vectors remaining: {zero_count} (was 1,562)')

# Show distribution by source
cur.execute("""
    SELECT e.dataset_source, COUNT(*)
    FROM cpe_vectors v
    JOIN cpe_entry e ON v.cpe_id = e.cpe_id
    WHERE v.tmd_dense IS NOT NULL
    GROUP BY e.dataset_source
    ORDER BY e.dataset_source
""")
print('\nDistribution by source:')
for dataset_source, count in cur.fetchall():
    source = dataset_source.split('_')[0] if dataset_source else None
    domain, task, modifier = TMD_MAP.get(source, TMD_MAP[None])
    print(f'  {dataset_source or "unknown":15s}: {count:4d} concepts -> domain={domain}, task={task}, modifier={modifier}')

cur.close()
conn.close()

print('\n✅ TMD fix complete!')
