# GWOM Ontology Traversal: Dataset Options for STEM & Programming

**Context**: P13 echo validation revealed 48.9% pass rate (below 80% threshold), indicating data quality issues with current FactoidWiki extractions. To improve training data quality for P15 LVM, we need high-quality **STEM and Programming-focused ontologies** with clean concept hierarchies.

**Recommendation**: Prioritize options that provide **propositional statements** and **logical relationships** over simple taxonomies.

---

## Top 5 Ontology Dataset Options

| # | Dataset | Size | Coverage | Quality | Access | Recommendation |
|---|---------|------|----------|---------|--------|----------------|
| 1 | **DBpedia Ontology (STEM subset)** | 4.58M concepts, ~500K STEM | Computer Science, Mathematics, Physics, Chemistry, Biology, Engineering | ⭐⭐⭐⭐⭐ High - Wikipedia-grounded | Open, SPARQL endpoint + dumps | **BEST - Start here** |
| 2 | **Wikidata (Science & Tech)** | 100M+ items, ~10M STEM | Comprehensive STEM coverage + programming languages, algorithms, data structures | ⭐⭐⭐⭐⭐ High - Community curated | Open, SPARQL + JSON dumps | **BEST - Complement DBpedia** |
| 3 | **ConceptNet (Technical subset)** | 21M edges, ~2M STEM-related | General + STEM common sense, programming concepts, algorithms | ⭐⭐⭐⭐ Good - Crowdsourced | Open, API + dumps | **GOOD - For common sense reasoning** |
| 4 | **Software Ontology (SWO)** | 5,000+ concepts | Software engineering, algorithms, data structures, design patterns | ⭐⭐⭐⭐⭐ Excellent - Expert curated | Open, OWL format | **EXCELLENT - Programming focus** |
| 5 | **Gene Ontology + Protein Ontology** | GO: 44K terms, PRO: 67K terms | Biology, biochemistry, molecular processes, cellular components | ⭐⭐⭐⭐⭐ Excellent - Scientific standard | Open, OBO format | **EXCELLENT - Biochemistry/Life sciences** |

---

## Detailed Analysis

### 1. DBpedia Ontology (STEM Subset) ⭐ RECOMMENDED

**Homepage**: https://www.dbpedia.org/
**Download**: https://databus.dbpedia.org/dbpedia/ontology
**Format**: RDF/OWL (can convert to JSON/JSONL)
**License**: CC BY-SA 3.0

#### Overview
DBpedia extracts structured data from Wikipedia, providing a rich ontology of concepts with typed relationships. The STEM subset includes computer science, mathematics, physics, chemistry, biology, and engineering domains.

#### Key Features
- **Propositional Knowledge**: Not just taxonomy - includes functional relationships
- **Example Concepts**:
  - `Algorithm → QuickSort → DivideAndConquerAlgorithm`
  - `Programming Language → Python → InterpretedLanguage`
  - `Chemical Compound → Glucose → Monosaccharide`
- **Relation Types**: `subClassOf`, `sameAs`, `relatedTo`, `isPartOf`, `causes`, `enables`
- **Coverage**: ~500K STEM concepts with rich metadata

#### Sample SPARQL Query (Technical Concepts)
```sparql
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbr: <http://dbpedia.org/resource/>

SELECT ?concept ?label ?parent ?description
WHERE {
  ?concept rdf:type/rdfs:subClassOf* dbo:Algorithm .
  ?concept rdfs:label ?label .
  ?concept dbo:abstract ?description .
  OPTIONAL { ?concept dbo:subClassOf ?parent }
  FILTER (lang(?label) = 'en')
  FILTER (lang(?description) = 'en')
}
LIMIT 1000
```

#### Why Use This
✅ **Propositions**: "QuickSort is a divide-and-conquer algorithm with O(n log n) average complexity"
✅ **Relationships**: Causal, temporal, hierarchical
✅ **Wikipedia-grounded**: High quality, human-verified
✅ **Large scale**: 500K STEM concepts = 50K+ ordered chains

#### Estimated GWOM Output
- **GraphRAG-style chains**: 30K (from internal CPESH → DBpedia links)
- **Ontology traversal chains**: 50K (pure DBpedia hierarchy walks)
- **Coherence**: Expected 0.85+ (high-quality Wikipedia text)

#### Integration Complexity
🟢 **Low** - SPARQL endpoint available, standard RDF format

---

### 2. Wikidata (Science & Technology) ⭐ RECOMMENDED

**Homepage**: https://www.wikidata.org/
**Download**: https://dumps.wikimedia.org/wikidatawiki/entities/
**Format**: JSON dumps, SPARQL endpoint
**License**: CC0 (public domain)

#### Overview
Wikidata is a free, collaborative knowledge base with 100M+ items. The Science & Technology subset includes comprehensive coverage of programming languages, algorithms, data structures, mathematical concepts, and scientific theories.

#### Key Features
- **Structured Propositions**: Subject-Predicate-Object triples with qualifiers
- **Example Items**:
  - `Q8028 (Algorithm)` → P31 (instance of) → Q11471 (concept in computer science)
  - `Q154755 (Bubble Sort)` → P31 → Q8028 → P366 (use: sorting)
  - `Q7397 (Software)` → P2283 (uses: programming language)
- **Relation Types**: 9,000+ property types including causal, functional, temporal
- **Coverage**: ~10M STEM items with rich metadata and cross-references

#### Sample SPARQL Query (Programming Concepts)
```sparql
SELECT ?item ?itemLabel ?description ?parent ?parentLabel
WHERE {
  ?item wdt:P31/wdt:P279* wd:Q8029 .  # Instance/subclass of "algorithm"
  ?item rdfs:label ?itemLabel .
  ?item schema:description ?description .
  OPTIONAL {
    ?item wdt:P279 ?parent .
    ?parent rdfs:label ?parentLabel .
  }
  FILTER(LANG(?itemLabel) = "en")
  FILTER(LANG(?description) = "en")
  FILTER(LANG(?parentLabel) = "en")
}
LIMIT 5000
```

#### Why Use This
✅ **Comprehensive**: 10M STEM items vs 500K in DBpedia
✅ **Structured Data**: Subject-Predicate-Object with qualifiers and references
✅ **Multilingual**: Can generate chains in multiple languages
✅ **Active Development**: Updated daily by community
✅ **Programming Focus**: 50K+ items on programming languages, frameworks, algorithms

#### Estimated GWOM Output
- **Ontology traversal chains**: 80K (largest source)
- **Cross-domain chains**: "Algorithm → Implementation → Programming Language → Runtime"
- **Coherence**: Expected 0.82+ (some noise from community edits)

#### Integration Complexity
🟡 **Medium** - Large JSON dumps (100+ GB), SPARQL timeout issues for complex queries

---

### 3. ConceptNet (Technical Subset)

**Homepage**: https://conceptnet.io/
**Download**: https://github.com/commonsense/conceptnet5/wiki/Downloads
**Format**: CSV/JSON edges
**License**: CC BY-SA 4.0

#### Overview
ConceptNet is a semantic network of common-sense knowledge with 21M edges. The technical subset includes programming concepts, algorithms, and STEM reasoning relationships.

#### Key Features
- **Common-Sense Reasoning**: "Sorting algorithm requires comparison", "Python is used for machine learning"
- **Example Relations**:
  - `/c/en/quicksort` → `IsA` → `/c/en/sorting_algorithm`
  - `/c/en/python` → `UsedFor` → `/c/en/machine_learning`
  - `/c/en/binary_search` → `Requires` → `/c/en/sorted_array`
- **Relation Types**: 36 types (IsA, UsedFor, Causes, HasPrerequisite, PartOf, etc.)
- **Coverage**: ~2M STEM-related concepts (out of 8M total)

#### Sample Query (Programming Concepts)
```python
import requests

# Get edges for "algorithm" concept
url = "http://api.conceptnet.io/query"
params = {
    "start": "/c/en/algorithm",
    "limit": 1000,
    "filter": "/c/en/"  # English only
}
response = requests.get(url, params=params)
edges = response.json()["edges"]

# Filter for technical relations
technical_relations = ["IsA", "UsedFor", "Causes", "HasPrerequisite", "PartOf"]
tech_edges = [e for e in edges if e["rel"]["label"] in technical_relations]
```

#### Why Use This
✅ **Common-Sense Links**: Fills gaps between formal ontology concepts
✅ **Procedural Knowledge**: "Hash table is used for fast lookup"
✅ **Crowdsourced**: Captures developer intuition and practical knowledge
✅ **API Access**: Easy integration without large downloads

#### Estimated GWOM Output
- **Ontology traversal chains**: 20K (medium scale)
- **Bridging chains**: Connects formal concepts with practical usage
- **Coherence**: Expected 0.75-0.80 (some noise from crowdsourcing)

#### Integration Complexity
🟢 **Low** - REST API, simple CSV/JSON format

---

### 4. Software Ontology (SWO) ⭐ PROGRAMMING FOCUS

**Homepage**: http://www.ebi.ac.uk/swo/
**Download**: https://github.com/allysonlister/swo
**Format**: OWL (Web Ontology Language)
**License**: CC BY 4.0

#### Overview
Software Ontology (SWO) is a resource for annotating software tools used in life sciences and bioinformatics, but includes comprehensive coverage of general software engineering concepts, algorithms, data structures, and design patterns.

#### Key Features
- **Expert-Curated**: Built by software engineering researchers and bioinformatics experts
- **Example Concepts**:
  - `Algorithm → SortingAlgorithm → ComparisonSort → QuickSort`
  - `DataStructure → Tree → BinaryTree → BinarySearchTree`
  - `DesignPattern → CreationalPattern → SingletonPattern`
- **Hierarchical Depth**: 5-8 levels deep (excellent for traversal)
- **Functional Annotations**: "QuickSort has average time complexity O(n log n)"

#### Sample OWL Concepts
```xml
<owl:Class rdf:about="http://www.ebi.ac.uk/swo/SWO_0000001">
  <rdfs:label>QuickSort Algorithm</rdfs:label>
  <rdfs:subClassOf rdf:resource="http://www.ebi.ac.uk/swo/SWO_0000100"/>
  <obo:IAO_0000115>A divide-and-conquer sorting algorithm...</obo:IAO_0000115>
  <swo:hasTimeComplexity>O(n log n)</swo:hasTimeComplexity>
</owl:Class>
```

#### Why Use This
✅ **Deep Hierarchies**: Perfect for ontology traversal (5-8 levels)
✅ **Programming-Centric**: Algorithms, data structures, design patterns
✅ **Formal Logic**: OWL format supports logical inference
✅ **Quality**: Expert-curated, no crowdsourcing noise
✅ **Educational**: Ideal for teaching/learning chains

#### Estimated GWOM Output
- **Ontology traversal chains**: 15K (smaller but highest quality)
- **Algorithm chains**: "Sorting → ComparisonSort → QuickSort → Lomuto Partition"
- **Coherence**: Expected 0.90+ (expert-curated)

#### Integration Complexity
🟡 **Medium** - OWL format requires parsing (use Owlready2 or rdflib), but file is small (~50MB)

---

### 5. Gene Ontology (GO) + Protein Ontology (PRO) ⭐ LIFE SCIENCES

**Homepage**: http://geneontology.org/ + https://proconsortium.org/
**Download**: http://current.geneontology.org/ontology/ + ftp://ftp.pir.georgetown.edu/databases/ontology/pro_obo/
**Format**: OBO (Open Biomedical Ontologies)
**License**: CC BY 4.0

#### Overview
Gold-standard ontologies for biological and biochemical concepts. GO covers molecular functions, biological processes, and cellular components. PRO covers proteins and their complexes.

#### Key Features
- **Scientific Standard**: Used by researchers worldwide for gene annotation
- **Example Concepts** (GO):
  - `BiologicalProcess → CellularMetabolicProcess → Photosynthesis → LightReaction`
  - `MolecularFunction → CatalyticActivity → Transferase → Kinase`
- **Example Concepts** (PRO):
  - `Protein → Enzyme → Kinase → ProteinKinase → SerineProteinKinase`
- **Hierarchical Depth**: 10-15 levels (excellent for deep traversal)
- **Logical Relationships**: `part_of`, `regulates`, `has_part`, `occurs_in`

#### Sample OBO Entry
```obo
[Term]
id: GO:0015979
name: photosynthesis
namespace: biological_process
def: "The synthesis by organisms of organic chemical compounds..."
is_a: GO:0019684 ! photosynthesis, light reaction
relationship: has_part GO:0009765 ! photosynthesis, light harvesting
```

#### Why Use This
✅ **Deep Hierarchies**: 10-15 levels = long ordered chains
✅ **Logical Rigor**: Formal relationships (part_of, regulates, etc.)
✅ **Scientific Accuracy**: Gold-standard for biology/biochemistry
✅ **Large Scale**: 44K GO terms + 67K PRO terms = 111K concepts
✅ **Fixes P13 Issues**: Biochemistry lanes had low scores (0.76) - this improves quality

#### Estimated GWOM Output
- **Ontology traversal chains**: 40K (large scale)
- **Biochemistry chains**: "Photosynthesis → LightReaction → Photolysis → OxygenProduction"
- **Coherence**: Expected 0.90+ (scientific standard)

#### Integration Complexity
🟢 **Low** - OBO format is simple text, well-documented parsers (pronto, fastobo)

---

## Recommended Implementation Strategy

### Phase 1: Quick Win (Week 1)
**Goal**: Generate 50K high-quality ontology chains

1. **Start with DBpedia** (30K chains)
   - Query SPARQL endpoint for STEM concepts
   - Export to JSONL: `{concept, parent, relations, description}`
   - Run breadth-first traversal (depth 3-5)

2. **Add Wikidata** (20K chains)
   - Download Science & Tech subset
   - Filter for programming + algorithms + mathematics
   - Run hierarchical traversal (depth 3-5)

**Estimated Time**: 2-3 days
**Expected Coherence**: 0.85+

---

### Phase 2: Programming Focus (Week 2)
**Goal**: Add 15K programming-specific chains

3. **Add Software Ontology (SWO)** (15K chains)
   - Download OWL file (~50MB)
   - Parse with Owlready2
   - Deep traversal (depth 5-8) for algorithms, data structures, design patterns

**Estimated Time**: 1-2 days
**Expected Coherence**: 0.90+

---

### Phase 3: Biochemistry Enhancement (Week 2)
**Goal**: Fix low biochemistry scores from P13

4. **Add Gene Ontology + Protein Ontology** (40K chains)
   - Download OBO files
   - Parse with pronto library
   - Deep traversal (depth 6-12) for biological processes

**Estimated Time**: 2-3 days
**Expected Coherence**: 0.90+

---

### Phase 4: Common Sense Bridging (Week 3)
**Goal**: Fill gaps with practical knowledge

5. **Add ConceptNet Technical Subset** (20K chains)
   - Query API for programming/STEM concepts
   - Filter for high-quality edges (confidence > 2.0)
   - Shallow traversal (depth 2-4)

**Estimated Time**: 1-2 days
**Expected Coherence**: 0.75-0.80

---

## Technical Implementation

### Data Format for GWOM Pipeline

```jsonl
{
  "seq_id": "ont-dbpedia-001",
  "method": "ontology",
  "source_ontology": "dbpedia",
  "concept_chain": [
    "Algorithm",
    "Sorting Algorithm",
    "Comparison Sort",
    "QuickSort",
    "Lomuto Partition Scheme"
  ],
  "concept_uris": [
    "http://dbpedia.org/resource/Algorithm",
    "http://dbpedia.org/resource/Sorting_algorithm",
    "http://dbpedia.org/resource/Comparison_sort",
    "http://dbpedia.org/resource/Quicksort",
    "http://dbpedia.org/resource/Lomuto_partition_scheme"
  ],
  "concept_descriptions": [
    "A finite sequence of well-defined instructions...",
    "An algorithm that sorts a list of elements...",
    "A sorting algorithm based on element comparisons...",
    "A divide-and-conquer sorting algorithm...",
    "A partition scheme for quicksort..."
  ],
  "relations": [
    {"from": 0, "to": 1, "type": "subClassOf"},
    {"from": 1, "to": 2, "type": "subClassOf"},
    {"from": 2, "to": 3, "type": "subClassOf"},
    {"from": 3, "to": 4, "type": "implements"}
  ],
  "quality_score": 0.92,
  "coherence_scores": [0.95, 0.93, 0.89, 0.91],
  "chain_length": 5,
  "max_depth": 4,
  "created_at": "2025-09-30T12:00:00Z"
}
```

### Code Skeleton

```python
# src/gwom/ontology_traverse.py

from rdflib import Graph
import requests

class OntologyTraverser:
    """Generate ordered concept chains from ontologies."""

    def __init__(self, source: str = "dbpedia"):
        self.source = source
        self.graph = None

    def fetch_dbpedia(self, root_concept: str, max_depth: int = 5):
        """Fetch DBpedia hierarchy via SPARQL."""
        query = f"""
        PREFIX dbo: <http://dbpedia.org/ontology/>
        SELECT ?concept ?label ?parent
        WHERE {{
          ?concept rdfs:subClassOf* dbo:{root_concept} .
          ?concept rdfs:label ?label .
          OPTIONAL {{ ?concept rdfs:subClassOf ?parent }}
          FILTER (lang(?label) = 'en')
        }}
        LIMIT 10000
        """
        # Execute SPARQL query
        # Build hierarchy
        # Return chains

    def traverse_hierarchy(self, root, max_depth=5, min_length=5):
        """Depth-first traversal to generate chains."""
        chains = []

        def dfs(node, path, depth):
            if depth > max_depth:
                if len(path) >= min_length:
                    chains.append(path.copy())
                return

            children = self.get_children(node)
            if not children:
                if len(path) >= min_length:
                    chains.append(path.copy())
                return

            for child in children:
                path.append(child)
                dfs(child, path, depth + 1)
                path.pop()

        dfs(root, [root], 0)
        return chains
```

---

## Storage & Index

### Ontology Cache Database

```sql
-- SQLite database: artifacts/gwom/ontology_cache.db

CREATE TABLE ontology_concepts (
    concept_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,  -- 'dbpedia' | 'wikidata' | 'conceptnet' | 'swo' | 'go'
    concept_uri TEXT UNIQUE NOT NULL,
    concept_label TEXT NOT NULL,
    concept_description TEXT,
    parent_id TEXT,
    depth INT,
    domain TEXT,  -- 'programming' | 'algorithms' | 'biochemistry' | 'mathematics'
    FOREIGN KEY (parent_id) REFERENCES ontology_concepts(concept_id)
);

CREATE INDEX idx_source ON ontology_concepts(source);
CREATE INDEX idx_domain ON ontology_concepts(domain);
CREATE INDEX idx_parent ON ontology_concepts(parent_id);

CREATE TABLE ontology_relations (
    relation_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,  -- 'subClassOf' | 'partOf' | 'causes' | etc.
    confidence REAL DEFAULT 1.0,
    FOREIGN KEY (source_id) REFERENCES ontology_concepts(concept_id),
    FOREIGN KEY (target_id) REFERENCES ontology_concepts(concept_id)
);
```

---

## Summary Table

| Dataset | GWOM Chains | Coherence | Complexity | Timeline | Priority |
|---------|-------------|-----------|------------|----------|----------|
| **DBpedia** | 30K | 0.85+ | 🟢 Low | 2-3 days | ⭐ P0 |
| **Wikidata** | 20K | 0.82+ | 🟡 Medium | 2-3 days | ⭐ P0 |
| **SWO** | 15K | 0.90+ | 🟡 Medium | 1-2 days | ⭐ P1 |
| **GO+PRO** | 40K | 0.90+ | 🟢 Low | 2-3 days | ⭐ P1 |
| **ConceptNet** | 20K | 0.75+ | 🟢 Low | 1-2 days | P2 |
| **TOTAL** | **125K** | **0.85 avg** | - | **8-13 days** | - |

---

## Next Steps

1. ✅ Review this document
2. ⏳ **Fix P13 low-quality entries** (257 missing negatives + malformed probes)
3. ⏳ **Start Phase 1** (DBpedia + Wikidata) to generate 50K ontology chains
4. ⏳ Test GWOM pipeline with 10K sample chains
5. ⏳ Scale to 125K total chains (all 5 sources)

---

**Document Status**: Draft v1.0
**Next Update**: After P13 data quality fixes
