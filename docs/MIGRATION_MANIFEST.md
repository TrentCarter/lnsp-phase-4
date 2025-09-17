# Migration Manifest for New Windsurf Project
## From: latent-neurolese-phase-3 → lnsp-phase-4

<!-- 20250917T0939_1 -->

---

## 🎯 CORE ARCHITECTURE FILES (Essential)

### Vec2Text Implementation (WORKING)
```
app/vect_text_vect/vec_text_vect_isolated.py  # ✅ PROVEN - Main isolated backend (0.68-0.98 cosine)
app/vect_text_vect/subscriber_wrappers/jxe_wrapper_proper.py  # JXE decoder wrapper
app/vect_text_vect/subscriber_wrappers/ielab_wrapper.py  # IELab decoder wrapper
app/vect_text_vect/subscriber_registry/  # Registry system for decoders
docs/how_to_use_jxe_and_ielab.md  # Critical usage documentation
```

### GTR-T5 Encoding/Decoding Pipeline
```
app/adapters/gtr_t5_encoder.py  # GTR-T5 encoder wrapper
app/cli/control_roundtrip.py  # ✅ WORKING roundtrip testing (Sep 12th)
working_dual_pipeline_test.py  # ✅ BREAKTHROUGH dual pipeline (Sep 10th)
tests/gtr_t5_cosine_roundtrip_test.py  # Cosine similarity validation
app/utils/download_models/download_gtr_t5_base.py  # Model downloader
```

---

## 📊 VALIDATED DATA (2.2M vectors, proven quality)

### Primary Vector Database
```
data/nemotron_vectors/tiny_stories/vectors_f32.npy  # 2.2M vectors (6.7GB)
data/curated_val.jsonl  # Curated validation set
```

### Downloaded Datasets
```
data/  # Contains Alpaca, GSM8K, Dolly, TinyStories, HellaSwag
scripts/download_lightweight_datasets.py  # Dataset downloader
scripts/download_nemotron_datasets.py  # Alternative datasets
scripts/convert_to_vectors.py  # Text → Vector conversion
```

---

## 🏗️ PROJECT CONFIGURATION

### Project Structure Files
```
inputs/projects/_project_config_file.json  # Base config template
inputs/projects/project_*  # Project-specific configs
app/utils/project_config.py  # Config loader
app/utils/vector_db_naming.py  # Naming conventions
```

### MLFlow Integration
```
app/utils/mlflow/mlflow_integration.py  # Core MLFlow integration
app/utils/mlflow/mlflow_service.py  # MLFlow service wrapper
app/utils/mlflow/project_to_mlflow.py  # Project → MLFlow converter
app/utils/mlflow/mlflow_dashboard.py  # Dashboard utilities
app/concepts/mlflow_tracker.py  # Concept tracking
```

---

## 🛠️ UTILITY SCRIPTS & TOOLS

### Data Generation & Processing
```
scripts/complete_concept_pipeline.py  # Full concept generation pipeline
scripts/qa_triplet_dataset_cli.py  # QA dataset generation
scripts/generate_adapter_training_data.py  # Adapter training data
app/utils/convert_vectors_to_f32.py  # Vector format conversion
```

### CLI Tools
```
app/utils/vec2text_cli.py  # Vec2Text CLI interface
app/utils/vmmoe_vec2text_batch_cli.py  # Batch processing
app/cli/eval_suite.py  # Evaluation suite
```

---

## ⚠️ LESSONS LEARNED (What NOT to carry over)

### Failed Approaches
- Most Mamba implementations in app/nemotron_vmmoe/ (mode collapse issues)
- Residual connections in Mamba (caused semantic copying)
- STELLA 1024D vectors (stick with GTR-T5 768D)
- Retrieval/similarity matching vec2text (use generative only)

### Working But Needs Improvement
- 4-word mechanical chunking (consider semantic concept segmentation)
- Current Mamba architecture (needs diversity loss to prevent collapse)

---

## 🔥 CRITICAL FILES TO MIGRATE FIRST

1. **Vec2Text System**
   - `app/vect_text_vect/vec_text_vect_isolated.py`
   - `docs/how_to_use_jxe_and_ielab.md`

2. **Data Pipeline**
   - `data/nemotron_vectors/tiny_stories/vectors_f32.npy`
   - `scripts/convert_to_vectors.py`
   - `app/adapters/gtr_t5_encoder.py`

3. **Project Configuration**
   - `inputs/projects/_project_config_file.json`
   - MLFlow integration files

4. **Working Tests**
   - `working_dual_pipeline_test.py`
   - `app/cli/control_roundtrip.py`

---

## 📝 NEW PROJECT REQUIREMENTS

### To Be Built Fresh
- FastAPI endpoints for text input/output
- FastAPI modular internal architecture
- LightRAG-type vector database with metadata
- Lightweight LLM for output refinement
- Iterative training system (1-10k batch updates)
- Real-time testing with scored responses DB

### To Keep From Current
- GTR-T5 768D encoding
- JXE/IELab vec2text decoding
- MLFlow experiment tracking
- Project config structure
- Input/output folder organization

---

## 🚀 MIGRATION STEPS

1. **Setup New Project Structure**
   ```bash
   mkdir -p /Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4/{app,data,inputs,output,scripts,tests,docs}
   mkdir -p /Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4/inputs/projects
   mkdir -p /Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4/app/{api,core,utils,adapters}
   ```

2. **Copy Core Files**
   ```bash
   # Copy vec2text system
   cp -r app/vect_text_vect /Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4/app/

   # Copy GTR-T5 components
   cp app/adapters/gtr_t5_encoder.py /Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4/app/adapters/

   # Copy validated data
   cp -r data/nemotron_vectors /Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4/data/
   cp data/curated_val.jsonl /Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4/data/

   # Copy project configs
   cp -r inputs/projects /Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4/inputs/

   # Copy MLFlow integration
   cp -r app/utils/mlflow /Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4/app/utils/

   # Copy working scripts
   cp scripts/convert_to_vectors.py /Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4/scripts/
   cp scripts/download_lightweight_datasets.py /Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4/scripts/
   ```

3. **Verify Critical Dependencies**
   ```bash
   # Check vec2text models are available
   python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('jxm/gtr-t5-base-multi-task-text')"
   ```

4. **Initialize New Components**
   - Setup FastAPI application structure
   - Configure LightRAG vector database
   - Implement iterative training pipeline
   - Add lightweight LLM for output refinement

---

## 📌 COMMANDS TO REMEMBER

```bash
# Vec2Text Usage (CRITICAL - always use isolated backend)
VEC2TEXT_FORCE_PROJECT_VENV=1 VEC2TEXT_DEVICE=cpu TOKENIZERS_PARALLELISM=false \
  ./venv/bin/python3 app/vect_text_vect/vec_text_vect_isolated.py \
  --input-text "YOUR TEXT" \
  --subscribers jxe,ielab \
  --vec2text-backend isolated \
  --output-format json \
  --steps 5

# Convert text to vectors
./.venv/bin/python3 scripts/convert_to_vectors.py

# Test roundtrip
TOKENIZERS_PARALLELISM=false ./.venv/bin/python3 -m app.cli.control_roundtrip \
  --curated data/curated_val.jsonl \
  --decoder-mode ielab \
  --gtr-model-name sentence-transformers/gtr-t5-base \
  --encoder-normalize true
```

---

## ✅ VALIDATION CHECKLIST

Before considering migration complete:
- [ ] Vec2text achieves 0.68+ cosine similarity
- [ ] GTR-T5 encoding/decoding roundtrip works
- [ ] MLFlow tracking is operational
- [ ] Project config system loads correctly
- [ ] FastAPI endpoints respond
- [ ] Vector database stores/retrieves successfully
- [ ] Can process new text → vector → text flow

---

## 📚 REFERENCE DOCS TO MIGRATE

```
docs/how_to_use_jxe_and_ielab.md  # CRITICAL
CLAUDE.md  # AI guidance (update for new project)
README.md  # Update for new architecture
```

---

## 🔧 VIRTUAL ENVIRONMENTS & REQUIREMENTS

### Environment Strategy
You have THREE options for vec2text setup:

1. **RECOMMENDED: Single Unified Environment**
   - Create one venv with all dependencies
   - Simpler to manage, less switching
   - Use the consolidated requirements below

2. **Separate Environments (if conflicts arise)**
   - `venv_jxe`: For JXE vec2text decoder
   - `venv_ielab`: For IELab vec2text decoder
   - Only needed if package conflicts occur

3. **Minimal Core + Docker**
   - Core dependencies in venv
   - Vec2text in Docker containers
   - Best for production isolation

### Core Requirements (requirements.txt)
```txt
# Core ML/AI
torch==2.1.2
transformers==4.36.2
sentence-transformers==3.0.1
vec2text==0.0.13
datasets==3.6.0
einops==0.8.1

# GTR-T5 and encoding
faiss-cpu>=1.8.0
numpy>=1.26,<2.0

# FastAPI for new architecture
fastapi==0.115.11
uvicorn==0.23.2
pydantic==2.10.6
pydantic-settings==2.9.1

# MLFlow tracking
mlflow>=2.18.0
SQLAlchemy==2.0.41
sqlmodel==0.0.24

# Utilities
tqdm==4.67.1
loguru==0.7.3
python-dotenv==1.1.0
rich==13.9.4
typer==0.15.2

# Data processing
pandas==2.2.3
pyarrow>=15.0.0,<19.0.0
scikit-learn==1.6.1

# Audio notifications (optional)
f5-tts-mlx==0.2.5
mlx==0.22.1
vocos-mlx==0.0.7
```

### Vec2Text Specific Requirements
```txt
# If using separate venv_jxe
vec2text==0.0.13
transformers==4.36.2  # Specific version for compatibility
torch==2.1.2
sentence-transformers==3.0.1

# If using separate venv_ielab
vec2text==0.0.13
transformers==4.36.2
torch==2.1.2
sentence-transformers==3.0.1
```

### Installation Commands
```bash
# Option 1: Single environment (RECOMMENDED)
cd /Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Verify vec2text installation
python -c "import vec2text; print(vec2text.__version__)"

# Download required models
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('jxm/gtr-t5-base-multi-task-text')"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/gtr-t5-base')"

# Option 2: Separate environments (if needed)
# For JXE
python3.11 -m venv venv_jxe
source venv_jxe/bin/activate
pip install vec2text==0.0.13 transformers==4.36.2 torch==2.1.2 sentence-transformers==3.0.1

# For IELab
python3.11 -m venv venv_ielab
source venv_ielab/bin/activate
pip install vec2text==0.0.13 transformers==4.36.2 torch==2.1.2 sentence-transformers==3.0.1
```

### Environment Variables (.env)
```bash
# Vec2Text settings
VEC2TEXT_FORCE_PROJECT_VENV=1
VEC2TEXT_DEVICE=cpu  # or cuda if available
TOKENIZERS_PARALLELISM=false
OMP_NUM_THREADS=1

# MLFlow
MLFLOW_TRACKING_URI=http://localhost:5007

# Model paths (optional, for caching)
HF_HOME=/Users/trentcarter/.cache/huggingface
TRANSFORMERS_CACHE=/Users/trentcarter/.cache/transformers
```

### Testing Vec2Text After Installation
```python
# test_vec2text_setup.py
import sys
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def test_setup():
    print("Testing vec2text setup...")

    # Test GTR-T5 encoder
    encoder = SentenceTransformer('sentence-transformers/gtr-t5-base')
    test_text = "Hello world"
    embedding = encoder.encode(test_text)
    print(f"✓ GTR-T5 encoding works: {embedding.shape}")

    # Test JXE model loading
    try:
        tokenizer = AutoTokenizer.from_pretrained("jxm/gtr-t5-base-multi-task-text")
        model = AutoModelForSeq2SeqLM.from_pretrained("jxm/gtr-t5-base-multi-task-text")
        print("✓ JXE vec2text models load successfully")
    except Exception as e:
        print(f"✗ JXE model loading failed: {e}")

    # Test IELab model loading
    try:
        tokenizer = AutoTokenizer.from_pretrained("ielab/gtr-t5-vec2text_gtr-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("ielab/gtr-t5-vec2text_gtr-base")
        print("✓ IELab vec2text models load successfully")
    except Exception as e:
        print(f"✗ IELab model loading failed: {e}")

    print("\nSetup verification complete!")

if __name__ == "__main__":
    test_setup()
```

### Notes on Virtual Environments

- **venv_jxe** and **venv_ielab** were originally created to isolate different vec2text decoder implementations
- In practice, both can coexist in a single environment with vec2text==0.0.13
- The separate environments are only needed if you encounter package conflicts
- The `VEC2TEXT_FORCE_PROJECT_VENV=1` environment variable ensures the correct venv is used

### Migration Decision Tree

```
Do you need immediate vec2text functionality?
├── YES → Use Option 1 (single unified venv)
└── NO → Start with core requirements, add vec2text later

Are you experiencing package conflicts?
├── YES → Use Option 2 (separate venv_jxe/venv_ielab)
└── NO → Continue with unified environment

Is this for production deployment?
├── YES → Consider Option 3 (Docker containers)
└── NO → Local venv is sufficient
```