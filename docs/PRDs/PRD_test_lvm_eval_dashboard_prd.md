# LVM Model Evaluation Dashboard - Product Requirements Document

## 1. Overview

The LVM Model Evaluation Dashboard is a web-based tool designed to facilitate the evaluation and comparison of Language-Vision Models (LVMs). It provides researchers and developers with an intuitive interface to test model performance across various metrics, visualize results, and maintain comprehensive evaluation logs.

## 2. Objectives

- Streamline the model evaluation workflow for LVM research
- Provide consistent and reproducible evaluation metrics
- Enable easy comparison between different model versions
- Support both in-distribution and out-of-distribution testing
- Maintain detailed logs of all evaluation runs

## 3. User Stories

### As a Researcher
- I want to quickly compare multiple model versions side by side
- I need to understand model performance on both IN and OOD data
- I want to visualize attention patterns and concept predictions
- I need to track model performance over time

### As a Developer
- I want an easy way to test new model checkpoints
- I need to understand model behavior across different metrics
- I want to identify performance regressions quickly
- I need to share evaluation results with my team

## 4. Features

### 4.1 Core Features

#### Model Management
- [x] List and filter available LVM models
- [x] Support for multiple model formats (.pt, .bin, etc.)
- [x] Model metadata display (creation date, size, architecture)
- [ ] Model version comparison

#### Evaluation Metrics
- [x] Next Concept Prediction accuracy
- [x] Cosine alignment scores
- [x] ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
- [x] Latency measurements
- [x] Memory usage tracking
- [ ] Perplexity scores
- [ ] BLEU scores

#### Testing Modes
- [x] IN-distribution testing
- [x] Out-of-distribution (OOD) testing
- [x] DIRECT pipeline testing (no LVM, GTR-T5→vec2text validation)
- [ ] Custom test set upload
- [ ] Cross-validation support

#### Test Data Sources
- [x] Wikipedia 790K dataset (790,391 chunks)
- [x] Complete sentences from Wikipedia articles
- [x] Real-world text reconstruction testing
- [ ] Custom ontology chains
- [ ] Domain-specific datasets

### 4.2 Advanced Features

#### Visualization
- [x] Interactive charts for metric comparison
- [x] Attention pattern visualization
- [ ] Gradient flow visualization
- [ ] Confusion matrices
- [ ] T-SNE/PCA projections

#### Analysis Tools
- [x] Concept prediction previews
- [ ] Error analysis tools
- [ ] Bias and fairness metrics
- [ ] Adversarial testing

#### Collaboration
- [ ] Shareable evaluation results
- [ ] Team annotations and notes
- [ ] Export to various formats (PDF, CSV, JSON)

## 5. Technical Specifications

### 5.1 System Architecture

```
lvm_eval/
├── lvm_dashboard.py      # Main application entry point
├── requirements.txt      # Python dependencies
├── static/               # Static files (CSS, JS, images)
│   ├── css/
│   ├── js/
│   └── favicon.ico
├── templates/            # HTML templates
│   └── index.html
├── logs/                 # Evaluation logs
├── routes.py             # Flask routes and evaluation logic
└── __init__.py           # Flask app initialization
```

### 5.2 Dataset Integration

#### Wikipedia 790K Dataset
The dashboard integrates with a massive Wikipedia dataset containing **790,391 chunks** of English Wikipedia content:

- **Raw Text Source**: `data/datasets/wikipedia/wikipedia_500k.jsonl` (2.1 GB)
  - 500,000 Wikipedia articles in JSONL format
  - Complete sentences and paragraphs, not single words or factoids
  - Each entry contains: id, title, text, url, and length fields

- **Vector Embeddings**: `artifacts/wikipedia_500k_corrected_vectors.npz` (2.1 GB)
  - 771,115 pre-computed 768-dimensional GTR-T5 embeddings
  - Used for similarity calculations and reconstruction testing

- **Database**: PostgreSQL `lnsp` database with 790,391 chunk entries
  - Structured storage for efficient retrieval
  - Metadata and relationship tracking

- **Training Data**: `artifacts/wikipedia_584k_fresh.npz`
  - P6 sequences for LVM model training
  - Sequential ontological relationships

#### DIRECT Pipeline Model
A special "DIRECT" model option that bypasses LVM processing:
- **Pipeline**: text → GTR-T5 (port 7001) → 768D → vec2text (port 7002) → text
- **Purpose**: Validate encoding/decoding infrastructure independently
- **Position**: Always appears at the top of the model list
- **Use Case**: Test the pipeline quality without LVM interference

#### Vector Sequence Processing Architecture
The dashboard implements proper sequence-based evaluation for next-chunk prediction:

**Input Processing:**
- **N sequential chunks** from Wikipedia articles (configurable, default N=5)
- Each chunk is **encoded individually** via GTR-T5 (port 7001)
- Results in a **sequence of N vectors** with shape `(N, 768)`
- Each vector represents a discrete **semantic concept/proposition**

**LVM Processing Flow:**
```
Chunk 1 → GTR-T5 → Vector 1 (768D) ┐
Chunk 2 → GTR-T5 → Vector 2 (768D) ├─→ Stack → (N, 768) tensor → LVM → Output (768D) → Vec2Text → Predicted Chunk N+1
Chunk 3 → GTR-T5 → Vector 3 (768D) │
...                                 │
Chunk N → GTR-T5 → Vector N (768D) ┘
```

**Key Characteristics:**
- LVMs process **sequences of concept vectors** similar to how LLMs process token sequences
- Each chunk loads as a **separate event** in the model's context window
- The model predicts the **next chunk** (N+1) given N input chunks
- Evaluation measures the similarity between predicted and actual next chunk

### 5.3 Data Model

#### Evaluation Run
```json
{
  "id": "eval_20251101_123456",
  "timestamp": "2025-11-01T12:34:56Z",
  "models": ["model1.pt", "model2.pt"],
  "test_mode": "both",
  "metrics": {
    "next_concept_prediction": true,
    "cosine_alignment": true,
    "rouge_scores": true
  },
  "results": {
    "model1.pt": {
      "next_concept_prediction": {
        "accuracy": 0.85,
        "loss": 0.15
      },
      "cosine_alignment": 0.92,
      "rouge_scores": {
        "rouge1": {"precision": 0.9, "recall": 0.88, "fmeasure": 0.89},
        "rouge2": {"precision": 0.85, "recall": 0.83, "fmeasure": 0.84},
        "rougeL": {"precision": 0.88, "recall": 0.86, "fmeasure": 0.87}
      },
      "latency_ms": 120.5,
      "memory_usage_mb": 2048.0
    }
  }
}
```

### 5.4 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main dashboard view |
| `/save_settings` | POST | Save user settings |
| `/evaluate` | POST | Run model evaluation (sequential) |
| `/evaluate/parallel` | POST | Run parallel model evaluation |
| `/evaluate/stream` | GET | SSE endpoint for progress updates |
| `/api/models` | GET | List available models (includes DIRECT) |
| `/api/system-info` | GET | Get system information |
| `/api/progress` | GET | Get current evaluation progress |

### 5.5 Service Dependencies

| Service | Port | Purpose |
|---------|------|---------|  
| GTR-T5 Encoder | 7001 | Text to 768D vector encoding |
| Vec2Text Decoder | 7002 | 768D vector to text decoding |
| PostgreSQL | 5432 | Wikipedia chunk storage (790K entries) |
| Dashboard | 8999 | LVM evaluation interface |

## 6. User Interface

### 6.1 Dashboard Layout

1. **Navigation Bar**
   - Application title
   - User profile/authentication
   - Help/documentation link

2. **Model Selection Panel (Left)**
   - Search and filter models
   - Model list with checkboxes
   - Evaluation settings
   - Start evaluation button

3. **Results Panel (Right)**
   - Interactive charts
   - Detailed metrics table
   - Concept predictions
   - Export options

### 6.2 Key Screens

1. **Model Selection View**
   - List of available models with checkboxes
   - Search and filter functionality
   - Model metadata preview

2. **Evaluation Results View**
   - Side-by-side model comparison
   - Interactive charts and graphs
   - Detailed metrics breakdown
   - Concept prediction visualization

3. **Settings View**
   - Default evaluation parameters
   - UI customization options
   - Data export settings
   - Vec2Text Decoding Steps control (persisted)
   - Starting Article index and Randomize start controls (persisted)
   - Metric selections (including Latency and Memory Usage) persisted

## 7. Performance Requirements

- **Response Time**: < 2s for model listing, < 30s for evaluation
- **Concurrent Users**: Support for 10+ simultaneous users
- **Data Retention**: 30 days of evaluation logs by default
- **Browser Support**: Chrome, Firefox, Safari (latest 2 versions)

## 8. Security Considerations

- Authentication and authorization for sensitive operations
- Input validation to prevent injection attacks
- Secure storage of model weights and evaluation data
- Rate limiting to prevent abuse

## 9. Future Enhancements

### Short-term (Next 3 months)
- [ ] Integration with experiment tracking tools (MLflow, Weights & Biases)
- [ ] Support for custom evaluation metrics
- [ ] Batch processing for large model sets

### Medium-term (3-6 months)
- [ ] Automated performance regression detection
- [ ] Integration with model training pipelines
- [ ] Advanced visualization tools

### Long-term (6+ months)
- [ ] Model fine-tuning capabilities
- [ ] Automated hyperparameter optimization
- [ ] Multi-modal evaluation support

## 10. Success Metrics

1. **Adoption Rate**: Number of active users
2. **Evaluation Volume**: Number of models evaluated per week
3. **Time Saved**: Reduction in time spent on manual evaluation
4. **Bug Detection**: Number of performance regressions caught early
5. **User Satisfaction**: Survey scores and feedback

## 11. Dependencies

### Software Dependencies
- Python 3.8+
- PyTorch 1.9.0+
- Flask 2.0.0+
- Transformers 4.11.0+
- scikit-learn 1.0.0+
- Chart.js 3.0.0+
- sentence-transformers
- rouge-score

### Data Dependencies
- **Wikipedia Dataset**: 790,391 chunks (not 80K as previously documented)
- **Vectors**: 771,115 pre-computed embeddings
- **Storage**: ~4.2 GB for JSONL + NPZ files
- **Memory**: 8+ GB RAM recommended for full dataset loading

## 12. Open Issues

1. **Performance Optimization**
   - Need to optimize model loading for faster evaluation
   - Consider model caching for repeated evaluations

2. **Feature Requests**
   - Support for custom test sets
   - Advanced visualization options
   - Integration with more experiment tracking tools

## 13. Appendix

### A. Glossary

- **LVM**: Language-Vision Model
- **IN-distribution**: Data similar to the training distribution
- **OOD**: Out-of-Distribution, data different from training distribution
- **ROUGE**: Recall-Oriented Understudy for Gisting Evaluation

### B. Related Documents

- [System Architecture](architecture.md)
- [API Documentation](api.md)
- [User Guide](user_guide.md)

### C. Changelog

#### 1.3.0 (2025-11-02 In Progress)
- Progress Completion UI: When evaluation completes, progress header switches to "Complete" and progress bar turns solid green (non-animated). New runs reset to blue striped/animated.
- Vec2Text Steps (UI + Backend): Added "Vec2Text Decoding Steps" input (persisted). Value is sent as `vec2text_steps` and used by decoder in `decode_vector(..., steps=...)`.
- Start Article Controls (UI + Backend): Added "Starting Article (index)" and "Randomize starting article" (persisted). Backend loads a window starting at `start_article_index` and shuffles when `random_start=true`.
- Metric Persistence: Latency and Memory Usage checkboxes (and all metrics) persist in localStorage and restore on load.
- Memory Usage Metric: Fixed negative values by reporting peak RSS delta over the evaluation (`peak_rss - initial_rss`), sampled after each test and on error path.
- Model Path Display: Results headers show project-relative paths (prefer `/lnsp-phase-4/…`, fallback to `/artifacts/…`, then basename). Model list shows relative path, timestamp, and size.
- Model Path Resolution (Backend): Paths like `/artifacts/...` are treated as project-root relative and resolved with `realpath()` before loading. Fixes "does not load" for UI-shown paths.
- Test Case Count: Route honors `num_test_cases` > 10; internal sample pool increased to support larger requests.
- DIRECT Handling: Expected output is last input chunk. Cosine compares encoded expected vs encoded decoded output text; guards for zero/NaN norms.
- UI State: Persist `numTestCases`, `numConcepts`, `testMode`, `selectAll`, `vec2textSteps`, `startArticleIndex`, and `randomStart`.

##### New/Updated Settings & Params
- UI Controls (persisted):
  - Vec2Text Decoding Steps (`#vec2textSteps`)
  - Starting Article (index) (`#startArticleIndex`)
  - Randomize starting article (`#randomStart`)
  - Metric checkboxes including Latency and Memory Usage (`.metric-checkbox`)
- Request JSON fields:
  - `vec2text_steps: int` — decoding steps for vec2text
  - `start_article_index: int` — starting article index (0-based)
  - `random_start: bool` — shuffle article window
  - `num_concepts: int`, `num_test_cases: int`, `test_mode: str`, `models: list[str]`, `metrics: list[str]`

##### Display & Pathing
- Model list shows relative path from project root, modified time, and size (MB).
- Results panel headers use project-relative paths for readability.

###### Example API Request Payload

```json
{
  "models": [
    "/artifacts/lvm/models/transformer_p5_20251102_095841/stageA/best_model.pt",
    "DIRECT"
  ],
  "test_mode": "both",
  "metrics": ["cosine", "rouge", "latency", "memory"],
  "num_concepts": 5,
  "num_test_cases": 25,
  "vec2text_steps": 5,
  "start_article_index": 250,
  "random_start": false
}
```

###### Example curl

```bash
curl -X POST http://localhost:8999/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "models": ["/artifacts/lvm/models/transformer_p5_20251102_095841/stageA/best_model.pt", "DIRECT"],
    "test_mode": "both",
    "metrics": ["cosine", "rouge", "latency", "memory"],
    "num_concepts": 5,
    "num_test_cases": 25,
    "vec2text_steps": 5,
    "start_article_index": 250,
    "random_start": false
  }'
```


#### 1.2.0 (2025-11-02)
- Renamed main entry point from `app.py` to `lvm_dashboard.py` for clarity
- Implemented proper vector sequence processing for next-chunk prediction
- Each input chunk now encoded separately into individual 768D vectors
- LVM processes sequences of N vectors (not concatenated text)
- Updated test data structure to maintain chunk separation

#### 1.1.0 (2025-11-02)
- Added DIRECT pipeline model for infrastructure testing
- Integrated Wikipedia 790K dataset (corrected from 80K)
- Fixed metric selection validation issues
- Improved vector-to-text decoding with port 7002
- Updated test data to use real Wikipedia chunks instead of synthetic data

#### 1.0.0 (2025-11-01)
- Initial release of LVM Model Evaluation Dashboard
- Core evaluation metrics and visualization
- Basic model management and comparison

### D. Quick Reference

#### Running the Dashboard
```bash
# Start the LVM evaluation dashboard
python lvm_eval/lvm_dashboard.py

# Dashboard will be available at:
# http://localhost:8999/
```

#### Dataset Commands
```bash
# Count Wikipedia chunks
wc -l data/datasets/wikipedia/wikipedia_500k.jsonl  # 500K articles

# Check vector dimensions  
python -c "import numpy as np; d=np.load('artifacts/wikipedia_500k_corrected_vectors.npz'); print(d['embeddings'].shape)"  # (771115, 768)
```

#### Testing the Pipeline
```bash
# Test DIRECT pipeline with 5 input chunks predicting the 6th
curl -X POST http://localhost:8999/evaluate -H "Content-Type: application/json" \
  -d '{"models":["DIRECT"],"test_mode":"both","num_concepts":5,"num_test_cases":10}'
```

---
*Document last updated: November 2, 2025*
