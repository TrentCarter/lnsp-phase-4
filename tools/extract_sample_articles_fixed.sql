-- Extract 4 Sample Wikipedia Articles with All Chunks
-- Purpose: Analyze temporal flow structure in actual article content

-- Step 1: Find 4 diverse articles (different sizes)
WITH article_stats AS (
    SELECT
        (chunk_position->>'article_index')::int AS article_id,
        chunk_position->>'article_title' AS title,
        COUNT(*) AS chunk_count,
        MIN((chunk_position->>'chunk_index')::int) AS first_chunk,
        MAX((chunk_position->>'chunk_index')::int) AS last_chunk
    FROM cpe_entry
    WHERE dataset_source = 'wikipedia_500k'
        AND chunk_position IS NOT NULL
        AND chunk_position->>'article_index' IS NOT NULL
    GROUP BY article_id, title
    HAVING COUNT(*) >= 10  -- At least 10 chunks
    ORDER BY chunk_count DESC
),
selected_articles AS (
    -- Pick 4 articles: 1 very long, 1 long, 1 medium, 1 short
    (SELECT * FROM article_stats WHERE chunk_count > 100 ORDER BY chunk_count DESC LIMIT 1)  -- Very long
    UNION ALL
    (SELECT * FROM article_stats WHERE chunk_count BETWEEN 50 AND 100 ORDER BY chunk_count DESC LIMIT 1)  -- Long
    UNION ALL
    (SELECT * FROM article_stats WHERE chunk_count BETWEEN 20 AND 50 ORDER BY chunk_count DESC LIMIT 1)  -- Medium
    UNION ALL
    (SELECT * FROM article_stats WHERE chunk_count BETWEEN 10 AND 20 ORDER BY chunk_count DESC LIMIT 1)  -- Short
)
SELECT
    sa.article_id,
    sa.title,
    sa.chunk_count,
    (ce.chunk_position->>'chunk_index')::int AS chunk_index,
    LEFT(ce.concept_text, 200) AS concept_preview,
    ce.cpe_id
FROM selected_articles sa
JOIN cpe_entry ce ON
    ce.chunk_position->>'article_index' = sa.article_id::text
    AND ce.dataset_source = 'wikipedia_500k'
ORDER BY
    sa.chunk_count DESC,
    sa.article_id,
    (ce.chunk_position->>'chunk_index')::int;
