# Sample Articles from Wikipedia 790K Dataset

**Date**: November 2, 2025
**Database**: PostgreSQL `lnsp` (790,391 total chunks)
**Purpose**: Analyze temporal flow structure in actual Wikipedia articles

---

## 📊 Selected Articles (4 Examples)

| Article ID | Title | Total Chunks | Range | Size Category |
|------------|-------|--------------|-------|---------------|
| **28** | **Apollo** | **1,107** | 0-1106 | Very Long (encyclopedic deity article) |
| **6942** | **Russian Orthodox bell ringing** | **100** | 0-99 | Long (specialized topic) |
| **13138** | **William West (Rhode Island politician)** | **50** | 0-49 | Medium (biographical) |
| **7628** | **Dennis Bock** | **20** | 0-19 | Short (concise biography) |

---

## 📍 Database Connection Details

```bash
# PostgreSQL Connection
Host: localhost
Port: 5432
Database: lnsp
User: lnsp
Password: lnsp

# Connection string
psql -h localhost -U lnsp -d lnsp
```

---

## 🔍 How to Query These Articles

### Get All Chunks for One Article

```sql
SELECT
    (chunk_position->>'chunk_index')::int AS chunk_num,
    concept_text,
    cpe_id
FROM cpe_entry
WHERE dataset_source = 'wikipedia_500k'
    AND chunk_position->>'article_index' = '28'  -- Apollo
ORDER BY (chunk_position->>'chunk_index')::int;
```

### Get Full Article Content (Python)

```python
import psycopg2

conn = psycopg2.connect(
    host='localhost',
    database='lnsp',
    user='lnsp',
    password='lnsp'
)

def get_article_chunks(article_id):
    cur = conn.cursor()
    query = """
        SELECT
            (chunk_position->>'chunk_index')::int AS chunk_idx,
            chunk_position->>'article_title' AS title,
            concept_text,
            cpe_id
        FROM cpe_entry
        WHERE dataset_source = 'wikipedia_500k'
            AND chunk_position->>'article_index' = %s
        ORDER BY (chunk_position->>'chunk_index')::int;
    """
    cur.execute(query, (str(article_id),))
    return cur.fetchall()

# Example: Get Apollo article
apollo_chunks = get_article_chunks(28)
print(f"Apollo has {len(apollo_chunks)} chunks")
for chunk_idx, title, text, cpe_id in apollo_chunks[:5]:
    print(f"Chunk {chunk_idx}: {text[:100]}...")
```

---

## 📋 Sample Content Preview

### Article 28: Apollo (1,107 chunks)

**Chunk 0** (Lead):
> "Apollo or Apollon is one of the Olympian deities in classical Greek and Roman religion and Greek and Roman mythology."

**Chunk 1** (Attributes):
> "Apollo has been recognized as a god of archery, music and dance, truth and prophecy, healing and diseases, the Sun and light, poetry, and more."

**Chunk 5** (Oracle):
> "As the patron deity of Delphi (Apollo Pythios), Apollo is an oracular god—the prophetic deity of the Delphic Oracle and also the deity of ritual purification."

**Chunk 27** (Etymology section begins):
> "Etymology Apollo (Attic, Ionic, and Homeric Greek: , ( ); Doric: , ; Arcadocypriot: , ; Aeolic: , ; ) The name Apollo—unlike the related older name Paean—is generally not found in the Linear B..."

**Temporal Flow Pattern**:
- Chunks 0-26: Introductory content, attributes, roles
- Chunk 27+: Etymology section (references back to "Apollo" name from lead)
- Later chunks: Detailed mythology, temples, worship practices (all reference earlier concepts)

---

## 🔬 Temporal Bias Analysis

### Expected Δ for These Articles

Based on Wikipedia temporal flow analysis (Δ = -0.0696 overall):

| Article | Expected Δ | Reason |
|---------|-----------|--------|
| **Apollo** | -0.08 to -0.10 | Large article, heavy cross-referencing, explanatory structure |
| **Russian Orthodox bell ringing** | -0.06 to -0.08 | Specialized topic, definitions → examples flow |
| **William West** | -0.05 to -0.07 | Biographical, chronological but with back-references |
| **Dennis Bock** | -0.03 to -0.05 | Short, less complex structure |

**Why all negative?**
- Lead introduces concepts (Apollo, deity, Greek, Roman)
- Later chunks reference these concepts repeatedly
- Example: Chunk 5 says "As patron deity of Delphi" (refers back to "deity" in chunk 0)
- Explanatory > Anticipatory structure

---

## 💾 Full Data Files

### Complete Article Data
- **Location**: `artifacts/lvm/sample_articles_full.txt`
- **Contains**: All chunks for all 4 articles with full text
- **Size**: ~1.5 MB
- **Format**: PostgreSQL table output

### Query Script
- **Location**: `tools/extract_sample_articles_fixed.sql`
- **Usage**:
  ```bash
  psql -h localhost -U lnsp -d lnsp -f tools/extract_sample_articles_fixed.sql
  ```

---

## 🎯 Use Cases for These Samples

### 1. Temporal Flow Analysis
- Manually read chunks 0-10 for each article
- Identify forward references (previews of upcoming content)
- Identify backward references (mentions of previous concepts)
- Calculate empirical Δ per article

### 2. Qualitative Examples for Reports
- Show real Wikipedia structure (not just statistics)
- Demonstrate backward bias with actual text
- Use in documentation/presentations

### 3. Test Data for Experiments
- Use Apollo (1107 chunks) for long-article testing
- Use Dennis Bock (20 chunks) for short-article baseline
- Compare Δ across article lengths

### 4. Training Data Filtering
- If implementing Δ-gating, use these as test cases
- Manually label forward/backward chunks
- Create gold standard for temporal flow detection

---

## 📊 Database Statistics

```sql
-- Total Wikipedia chunks in database
SELECT COUNT(*) FROM cpe_entry WHERE dataset_source = 'wikipedia_500k';
-- Result: 790,391

-- Article count distribution
SELECT
    CASE
        WHEN chunk_count > 100 THEN '>100 chunks (very long)'
        WHEN chunk_count > 50 THEN '50-100 chunks (long)'
        WHEN chunk_count > 20 THEN '20-50 chunks (medium)'
        WHEN chunk_count > 10 THEN '10-20 chunks (short)'
        ELSE '<10 chunks (very short)'
    END AS size_category,
    COUNT(*) as article_count
FROM (
    SELECT COUNT(*) as chunk_count
    FROM cpe_entry
    WHERE dataset_source = 'wikipedia_500k'
    GROUP BY chunk_position->>'article_index'
) article_sizes
GROUP BY size_category
ORDER BY article_count DESC;
```

---

## 🔗 Related Files

- **Wikipedia Dataset Guide**: `docs/WIKIPEDIA_790K_DATASET_GUIDE.md`
- **Temporal Flow Analysis**: `artifacts/lvm/wikipedia_temporal_analysis/REPORT.md`
- **Database Locations**: `docs/DATABASE_LOCATIONS.md`
- **Full Query Results**: `artifacts/lvm/sample_articles_full.txt`

---

**Last Updated**: November 2, 2025
**Dataset Version**: Wikipedia 790K (500k articles → 790,391 chunks)
