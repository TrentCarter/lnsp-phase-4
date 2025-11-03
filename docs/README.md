# LNSP Phase 4 Documentation Index

## 📚 Core Documentation

### Dataset Documentation
- **[Wikipedia 790K Dataset Complete Guide](./WIKIPEDIA_790K_DATASET_GUIDE.md)** - Comprehensive reference for the 790,391 Wikipedia chunks dataset
- **[Wikipedia Dataset Quick Reference](./WIKIPEDIA_DATASET_QUICK_REFERENCE.md)** - Quick lookup for dataset locations and commands
- **[Chunking Documentation](../CHUNKING_README.md)** - Text chunking pipeline documentation

### System Architecture
- **[FastAPI Services PRD](./PRDs/PRD_FastAPI_Services.md)** - Product requirements for API services
- **[LVM Evaluation Dashboard PRD](./PRDs/PRD_test_lvm_eval_dashboard_prd.md)** - Dashboard specifications
- **[Retrieval System Index](./RETRIEVAL_DOCS_INDEX.md)** - Vector retrieval documentation

### Model Documentation
- **[AMN Model Card](./ModelCards/AMN_v0.md)** - Adaptive Memory Network specifications
- **[GRU Model Card](./ModelCards/GRU_v0.md)** - GRU model specifications
- **[Training Strategy](./Iterative_Training_Strategy.md)** - Iterative training approach

### Development Guides
- **[Development Summary](./Development_Summary.md)** - Project development overview
- **[Future Development Plan](./Future_Development_Plan.md)** - Roadmap and next steps
- **[Production Readiness](../PRODUCTION_READINESS_FIXES.md)** - Production deployment checklist

## 🚀 Quick Start Guides

### For New Developers
1. Start with [Wikipedia Dataset Quick Reference](./WIKIPEDIA_DATASET_QUICK_REFERENCE.md)
2. Review [Development Summary](./Development_Summary.md)
3. Check [FastAPI Services PRD](./PRDs/PRD_FastAPI_Services.md)

### For Data Scientists
1. Read [Wikipedia 790K Dataset Guide](./WIKIPEDIA_790K_DATASET_GUIDE.md)
2. Review [Model Cards](./ModelCards/)
3. Study [Training Strategy](./Iterative_Training_Strategy.md)

### For DevOps
1. Check [Production Readiness](../PRODUCTION_READINESS_FIXES.md)
2. Review [Rollout Plan](../PRODUCTION_ROLLOUT_PLAN.md)
3. See deployment scripts in `/scripts/`

## 📊 Key Dataset Numbers

- **790,391** - Total Wikipedia chunks in PostgreSQL
- **771,115** - Available vector embeddings
- **500,000** - Raw Wikipedia articles in JSONL
- **768** - Embedding dimensions
- **2.1 GB** - Size of both JSONL and NPZ files

## 🔧 Service Endpoints

| Service | Port | Purpose |
|---------|------|---------|
| GTR-T5 Encoder | 7001 | Text to vectors |
| Vec2Text Decoder | 7002 | Vectors to text |
| LVM Dashboard | 8999 | Model evaluation |
| PostgreSQL | 5432 | Database |
| MLFlow | 5007 | Experiment tracking |

## 📁 Project Structure

```
lnsp-phase-4/
├── docs/                  # This directory
│   ├── WIKIPEDIA_790K_DATASET_GUIDE.md
│   ├── WIKIPEDIA_DATASET_QUICK_REFERENCE.md
│   ├── PRDs/             # Product requirement docs
│   └── ModelCards/       # Model specifications
├── data/
│   └── datasets/
│       └── wikipedia/    # 500k Wikipedia JSONL
├── artifacts/
│   ├── wikipedia_500k_corrected_vectors.npz
│   └── wikipedia_584k_fresh.npz
├── lvm_eval/            # Evaluation dashboard
├── app/                 # Core application
└── tools/              # Utility scripts
```

## 🔄 Recent Updates

- **Nov 2, 2024**: Added Wikipedia 790K dataset documentation
- **Nov 2, 2024**: Implemented DIRECT pipeline model for testing
- **Nov 2, 2024**: Fixed LVM evaluation dashboard issues
- **Oct 2024**: Completed P6b training with 584K sequences

## 📝 Contributing

When adding new documentation:
1. Place it in the appropriate subdirectory
2. Update this README index
3. Use clear, descriptive filenames
4. Include creation/update dates
5. Cross-reference related documents

---
*Last Updated: November 2, 2024*
