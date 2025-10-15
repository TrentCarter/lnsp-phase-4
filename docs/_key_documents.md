# 📚 Key Documents Index

**Last Updated**: October 12, 2025 - LVM Testing Complete ✅

---

## 🔥 START HERE - Latest Session

**SESSION_HANDOFF_OCT12_LVM_COMPLETE.md** ⭐⭐⭐
- Complete summary of LVM testing (all 3 phases)
- Final results: GRU recommended for production
- Next steps: Deploy to LVM server (port 8003)
- **Read this first for context on latest work**

---

## 🎯 LVM (Latent Vector Models) - PRODUCTION READY

**Status**: All 3 test phases complete, ready for deployment

### Quick Reference
1. **SESSION_HANDOFF_OCT12_LVM_COMPLETE.md** - Session summary & next steps
2. **LVM_COMPREHENSIVE_TEST_RESULTS.md** - Detailed analysis with charts
3. **LVM_TEST_SUMMARY_FINAL.md** - Executive summary
4. **LVM_TRAINING_RESULTS_OCT12.md** - Training session details

### Official Documentation
5. **docs/PRDs/PRD_LVM_Test_Results.md** - Official test results
6. **docs/PRDs/PRD_LVM_Models.md** - Architecture tracking matrix
7. **docs/PRDs/PRD_LVM_Test_Plan.md** - 4-phase testing strategy
8. **docs/PRDs/PRD_LVM_Flowchart.md** - Test flow diagrams

### Key Results
- ✅ **Phase 1**: Model validation (78.30-78.60% cosine)
- ✅ **Phase 2**: Top-K retrieval (500-1,520x better than random)
- ✅ **Phase 3**: Vec2text integration (100% success, 75-77% reconstruction)
- 🏆 **Winner**: GRU model (best balance: 76.90% vec2text, 29.76% Top-20)

### Model Checkpoints
- `artifacts/lvm/models/lstm_baseline/best_model.pt` (5.1M params)
- `artifacts/lvm/models/mamba2/best_model.pt` (7.1M params) ⭐ **RECOMMENDED**
- `artifacts/lvm/models/transformer/best_model.pt` (17.6M params)

### Test Data
- `artifacts/lvm/evaluation/phase1_test_results.json`
- `artifacts/lvm/evaluation/phase2_retrieval_results.json`
- `artifacts/lvm/evaluation/phase3_vec2text_results.json`

---

## 🏗️ Architecture & System Design

### Core Architecture
1. **LNSP_LONG_TERM_MEMORY.md** - Cardinal rules (NEVER violate!)
2. **CLAUDE.md** - Daily operational guidelines
3. **docs/TOKENLESS_MAMBA_ARCHITECTURE.md** - Original LVM design
4. **docs/LVM_ARCHITECTURE_OPTIONS.md** - 12 architectures + 100-iteration plan

### FastAPI Services
5. **docs/PRDs/PRD_FastAPI_Services.md** - Microservices architecture
   - Port 8001: Chunker
   - Port 8002: TMD Router
   - Port 8003: LVM Server (ready for GRU deployment)
   - Port 8765: GTR-T5 Embeddings
   - Port 8766: Vec2Text Decoder (fixed!)

### TMD-LS (Task-Modifier-Domain Lane Specialists)
6. **docs/TMD-LS_IMPLEMENTATION_COMPLETE.md** - Multi-LLM routing
7. **configs/llm_prompts/llm_prompts_master.json** - Prompt templates

---

## 📊 Data & Datasets

### Requirements & Validation
1. **docs/PRDs/PRD_Dataset_Requirements.md** - Data specifications
2. **docs/PRDs/PRD_Sequential_Training_Data.md** - LVM training data

### Chunking & Processing
3. **CHUNKING_API_COMPLETE_GUIDE.md** - Complete chunking guide
4. **CHUNKING_QUICK_REFERENCE.md** - Quick reference
5. **docs/CHUNKING_PARAMETERS_GUIDE.md** - Parameter tuning

### Wikipedia Pipeline
6. **docs/WIKIPEDIA_PIPELINE_SUMMARY.md** - Large-scale ingestion
7. **docs/FINAL_PLAN_OCT11_SEQUENTIAL_DATA.md** - Sequential data generation

---

## 🔧 Development & Operations

### How-To Guides
1. **docs/how_to_use_jxe_and_ielab.md** - Vec2text usage (JXE/IELab)
2. **docs/howto/how_to_access_local_AI.md** - Ollama + Llama 3.1 setup

### Optimization & Performance
3. **docs/OPTIMIZATION_SUMMARY_TABLE.md** - Performance optimizations
4. **docs/OPTIMIZATION_TMD_PASSTHROUGH_OCT12.md** - TMD passthrough
5. **docs/BATCH_EMBEDDINGS_ARCHITECTURE.md** - Batch processing

### Databases & Storage
6. **docs/DATABASE_LOCATIONS.md** - PostgreSQL, Neo4j, FAISS locations
7. **docs/DATA_VERIFICATION_OCT12.md** - Data integrity checks

---

## 🧪 Testing & Validation

### LVM Test Scripts (Latest)
- `test_phase1_standalone.py` - Model loading, validation, speed
- `test_phase2_retrieval.py` - Top-K retrieval with FAISS
- `test_phase3_vec2text.py` - Vec2text integration

### Test Results
- **Phase 1**: All models pass (78.30-78.60% cosine, <1.2ms/batch)
- **Phase 2**: 23.44-36.17% Top-20 accuracy (500-1,520x > random)
- **Phase 3**: 100% decoding success, 75.31-76.90% reconstruction

---

## 📝 Session Summaries (Historical)

1. **SESSION_HANDOFF_OCT12_LVM_COMPLETE.md** - LVM testing complete ✅
2. **SESSION_HANDOFF_OCT12_LVM_READY.md** - Pre-testing setup
3. **SESSION_SUMMARY_OCT11_SEQUENTIAL_DATA.md** - Sequential data generation
4. **SESSION_SUMMARY_OCT12_TMD_OPTIMIZATION.md** - TMD passthrough
5. **docs/SESSION_SUMMARY_OCT9_DATABASES_FASTAPI.md** - Database setup
6. **SESSION_SUMMARY_OCT8_CHUNKING.md** - Chunking implementation

---

## 🚀 Quick Commands for Next Session

### View Latest Results
```bash
# Read session handoff
cat SESSION_HANDOFF_OCT12_LVM_COMPLETE.md

# Check test results
cat artifacts/lvm/evaluation/phase3_vec2text_results.json

# View model checkpoints
ls -lh artifacts/lvm/models/*/best_model.pt
```

### Deploy GRU to Production
```bash
# Update app/api/lvm_server.py to load GRU checkpoint
# Then start server:
./.venv/bin/uvicorn app.api.lvm_server:app --host 127.0.0.1 --port 8003
```

### Rerun Tests (if needed)
```bash
# Phase 1: Model validation
./.venv/bin/python test_phase1_standalone.py

# Phase 2: Top-K retrieval
./.venv/bin/python test_phase2_retrieval.py

# Phase 3: Vec2text integration
nohup ./.venv/bin/uvicorn app.api.vec2text_server:app --host 127.0.0.1 --port 8766 > /tmp/vec2text_server.log 2>&1 &
./.venv/bin/python test_phase3_vec2text.py
```

---

## ✅ Current Status

**LVM System**: ✅ PRODUCTION READY
- 3 models trained and validated
- All tests passed (7/7 criteria met)
- GRU model recommended for deployment
- Ready for integration with TMD-LS pipeline

**Next Milestone**: Deploy GRU to LVM server and test end-to-end pipeline

---

**Document Version**: 3.0 (Oct 12, 2025)
**Maintained by**: Claude Code sessions
**Ready for**: `/clear` and next session
