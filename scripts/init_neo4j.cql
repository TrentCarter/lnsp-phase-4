// FactoidWiki → LNSP Neo4j bootstrap schema
// Version: 0.1  (2025-09-21)
// Idempotent: safe to re-run

// Constraints -----------------------------------------------------
CREATE CONSTRAINT concept_id IF NOT EXISTS
FOR (c:Concept) REQUIRE c.cpe_id IS UNIQUE;

// Indexes ---------------------------------------------------------
CREATE INDEX concept_lane_idx IF NOT EXISTS
FOR (c:Concept) ON (c.laneIndex);

CREATE INDEX concept_domain_idx IF NOT EXISTS
FOR (c:Concept) ON (c.domainCode);

CREATE INDEX concept_task_idx IF NOT EXISTS
FOR (c:Concept) ON (c.taskCode);

CREATE INDEX concept_modifier_idx IF NOT EXISTS
FOR (c:Concept) ON (c.modifierCode);

// Sample upsert procedure (optional)
// CALL {
//   MERGE (c:Concept {cpe_id:$cpe_id})
//   SET c.text=$concept_text,
//       c.tmdBits=$tmd_bits,
//       c.tmdLane=$tmd_lane,
//       c.laneIndex=$lane_index,
//       c.domainCode=$domain_code,
//       c.taskCode=$task_code,
//       c.modifierCode=$modifier_code,
//       c.echoScore=$echo_score,
//       c.validationStatus=$validation_status
//   RETURN c
// };

// Edge creation example (executed from app loader)
// MATCH (src:Concept {cpe_id:$src}), (dst:Concept {cpe_id:$dst})
// MERGE (src)-[r:REL {type:$rel_type}]->(dst)
// SET r.confidence=$confidence,
//     r.properties=$properties;

// Tips ------------------------------------------------------------
// To run:
//   cypher-shell -u neo4j -p password -f scripts/init_neo4j.cql
// or use the Neo4j Browser to paste this script.
