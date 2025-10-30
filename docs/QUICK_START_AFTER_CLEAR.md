# 🚀 Quick Start After /clear

**Last Updated:** October 29, 2025
**Purpose:** Fast orientation after conversation clear

---

## ✅ CURRENT SYSTEM STATUS (October 29, 2025)

**All systems operational. LSTM bug fixed today.**

### Production Services Running

| Port | Service | Status | Purpose |
|------|---------|--------|---------|
| 9000 | Master Chat | ✅ | UI for all models |
| 9001 | AMN | ✅ | Fast inference (0.49ms) |
| 9002 | Transformer Baseline | ✅ | Research |
| 9003 | GRU | ✅ | Best accuracy (0.5920) |
| 9004 | LSTM ✅ | ✅ | **FIXED TODAY** (0.5792) |
| 9005 | Vec2Text Direct | ✅ | Best quality |
| 9006 | Transformer Optimized | ✅ | Research |

**Health Check:** `curl http://localhost:900X/health`

---

## 🎉 WHAT HAPPENED TODAY (Oct 29, 2025)

### LSTM Bug Fixed! ✅

**Problem:** LSTM model serving random weights (0.4102 val cosine)

**Root Cause:** Architecture mismatch between training checkpoint (4-stack residual blocks) and inference code (simple 2-layer LSTM)

**Fix:**
1. Updated `app/lvm/model.py` with correct architecture
2. Retrained model
3. Replaced `artifacts/lvm/models/lstm_v0.pt`
4. Restarted service on port 9004

**Results:**
- In-Dist: 0.4102 → 0.5792 (+41%)
- OOD: 0.4427 → 0.6046 (+37%)
- Status: BROKEN → PRODUCTION READY

**See:** `docs/LSTM_FIX_SUMMARY_OCT29_2025.md` for full details

---

## 📊 LVM MODEL RANKINGS (All 5 Production-Ready)

| Rank | Model | In-Dist | OOD | Latency | Best For |
|------|-------|---------|-----|---------|----------|
| 🥇 | AMN | 0.5597 | 0.6375 | 0.49ms ⚡ | Fast + Best OOD |
| 🥈 | GRU | 0.5920 | 0.6295 | 2.11ms | Best Accuracy |
| 🥉 | Transformer | 0.5864 | 0.6257 | 2.65ms | Research |
| 4 | LSTM ✅ | 0.5792 | 0.6046 | 0.56ms ⚡ | Balanced (Fixed today) |

---

## 📁 KEY FILES TO READ FIRST

**System Status:**
- `docs/SYSTEM_STATE_SUMMARY.md` - Complete system overview with 3 strategic suggestions
- `docs/DATABASE_LOCATIONS.md` - Where all data lives (databases, vectors, models)

**LVM Models:**
- `artifacts/lvm/COMPREHENSIVE_LEADERBOARD.md` - Complete model benchmarks + LSTM fix details
- `docs/LVM_DATA_MAP.md` - Training data, model locations, inference pipeline

**Recent Work:**
- `docs/LSTM_FIX_SUMMARY_OCT29_2025.md` - Today's LSTM fix (complete documentation)

**Project Guide:**
- `CLAUDE.md` - Instructions for Claude Code (always read this first!)

---

## 🎯 CURRENT PRIORITIES

1. ~~**Fix LSTM Bug**~~ ✅ **COMPLETED TODAY** (Oct 29, 2025)
   - All 5 LVM models now production-ready

2. **Vec2Text Bottleneck** 🔴 **CRITICAL NEXT STEP**
   - Current: 900-6200ms (90-98% of total latency)
   - Target: <100ms with hybrid vocabulary decoder
   - Options:
     - Hybrid approach: Fast vocab decoder + vec2text fallback (2-3 days)
     - Full replacement: End-to-end vocab decoder (3-5 days)
   - **See:** `docs/SYSTEM_STATE_SUMMARY.md` Part 4, Suggestion 2 for full plan

3. **Production Deployment** 🟡 MEDIUM
   - Load balancing, failover, monitoring
   - 4-week timeline
   - **See:** `docs/SYSTEM_STATE_SUMMARY.md` Part 4, Suggestion 3

---

## 🔧 QUICK COMMANDS

### Check Service Health
```bash
# All services
for port in {9000..9006}; do
  echo -n "Port $port: "
  curl -s http://localhost:$port/health | grep -o '"status":"[^"]*"' || echo "down"
done
```

### Test LSTM Model (Fixed Today)
```bash
curl -X POST http://localhost:9004/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": ["What is diabetes?"], "conversation_id": "test", "decode_steps": 1}'
```

### Run OOD Evaluation
```bash
./.venv/bin/python tools/eval_lstm_ood.py
```

### Start/Stop Services
```bash
./scripts/start_lvm_services.sh
./scripts/stop_lvm_services.sh
```

---

## 🚨 WHAT NOT TO DO

1. **DON'T retrain ontology data for LVM models**
   - Ontologies (WordNet, SWO, GO) are taxonomic, not sequential
   - Only use Wikipedia, papers, stories for LVM training
   - **See:** CLAUDE.md section "CRITICAL: NEVER USE ONTOLOGICAL DATASETS FOR LVM TRAINING"

2. **DON'T use sentence-transformers directly for vec2text**
   - 9.9x worse quality than correct encoder
   - Always use `IsolatedVecTextVectOrchestrator`
   - **See:** `docs/how_to_use_jxe_and_ielab.md`

3. **DON'T commit without explicit permission**
   - Ask before creating commits
   - **See:** CLAUDE.md Git Safety Protocol

4. **DON'T skip the KMP_DUPLICATE_LIB_OK fix on macOS**
   - Required for training to prevent "Abort trap: 6" crashes
   - Always set: `export KMP_DUPLICATE_LIB_OK=TRUE`
   - **See:** CLAUDE.md macOS OpenMP Fix section

---

## 📞 WHERE TO FIND THINGS

**Code:**
- LVM models: `app/lvm/model.py`
- FastAPI services: `app/api/lvm_inference.py`, `app/api/master_chat.py`
- Training scripts: `tools/train_*.py`

**Data:**
- Model checkpoints: `artifacts/lvm/models/`
- Training sequences: `artifacts/lvm/training_sequences_ctx5.npz`
- OOD test data: `artifacts/lvm/wikipedia_ood_test_ctx5.npz`
- Wikipedia vectors: `artifacts/wikipedia_500k_corrected_vectors.npz`

**Documentation:**
- System status: `docs/SYSTEM_STATE_SUMMARY.md`
- Database locations: `docs/DATABASE_LOCATIONS.md`
- Data flow: `docs/DATA_FLOW_DIAGRAM.md`
- Leaderboard: `artifacts/lvm/COMPREHENSIVE_LEADERBOARD.md`

**Recent Work:**
- LSTM fix: `docs/LSTM_FIX_SUMMARY_OCT29_2025.md`
- Evaluation script: `tools/eval_lstm_ood.py`

---

## 🔮 IMMEDIATE NEXT ACTIONS

If continuing from today's work:

1. **Monitor LSTM in production** (optional)
   - Check port 9004 health periodically
   - Watch for any issues with real traffic
   - Verify latency stays ~0.56ms

2. **Start vec2text bottleneck work** (recommended)
   - Review `docs/SYSTEM_STATE_SUMMARY.md` Part 4, Suggestion 2
   - Decide: Hybrid approach or full replacement?
   - Create implementation plan
   - Start with vocabulary decoder prototype

3. **Update service auto-start scripts** (cleanup)
   - Ensure `scripts/start_lvm_services.sh` uses correct model paths
   - Add health check validation
   - Document startup sequence

---

**Remember:** All 5 LVM models are now production-ready as of October 29, 2025! 🎉

**Next big win:** Solve the vec2text bottleneck (50-1000x speedup possible)

---

**Quick Links:**
- System Overview: `docs/SYSTEM_STATE_SUMMARY.md`
- LSTM Fix Details: `docs/LSTM_FIX_SUMMARY_OCT29_2025.md`
- Model Leaderboard: `artifacts/lvm/COMPREHENSIVE_LEADERBOARD.md`
- Project Instructions: `CLAUDE.md`
