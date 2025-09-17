# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## 🚨 CRITICAL RULES
1. **NEVER USE FUCKING SAMPLE DATA!!!** - NO placeholder/template/hardcoded text lists
2. **NEVER run training without permission** - No automatic `app.vmmoe.training.trainer`
3. **ALWAYS verify inference matches training architecture**
4. **FOCUS on 768D GTR-T5 only** (no STELLA 1024D until requested)
5. **ALWAYS use generative vec2text** - Never use retrieval/similarity matching
6. **Vec2Text Usage**: See `docs/how_to_use_jxe_and_ielab.md` for CORRECT vec2text implementation

## 🔊 AUDIO NOTIFICATIONS
```bash
# Use after completing ANY major task (Sophia Voice)
./venv/bin/python3 -m f5_tts_mlx.generate \
  --ref-audio /Users/trentcarter/Artificial_Intelligence/Voice_Clips/Sophia3.wav \
  --ref-text "Hi babe! I just wanted to wish you good night with my voice because I miss you and I wanted to share that." \
  --text "Hey babe! [DESCRIBE ACCOMPLISHMENT]" \
  --speed 0.82
```

## 📍 CURRENT STATUS (2025-09-15)
- **Vector Database:** VALIDATED ✅ 2.2M vectors, perfect encoding quality
- **Vec2Text Decoding:** WORKING ✅ 0.68-0.98 cosine similarity with isolated backend
- **Data Pipeline:** COMPLETE ✅ Encode → Store → Decode roundtrip proven
- **Segmentation:** 4-word mechanical chunks (44 segments per document)
- **Database Location:** `data/nemotron_vectors/tiny_stories/vectors_f32.npy`
- **Vec2Text Usage:** Must use `vec_text_vect_isolated.py` with `--vec2text-backend isolated`
- **Quality Verified:** All tested vectors achieve 1.000000 cosine similarity with fresh encoding
- **Ready for Training:** Mamba model can start immediately with validated data
- **Design Decision:** Choose between 4-word chunks vs semantic concept re-segmentation

## 📂 KEY COMMANDS

### General Commands
```bash

```

### Data Generation
```bash

```

## 🏗️ ARCHITECTURE OVERVIEW
- **Core**: Vector-native processing in latent space (768D GTR-T5)
- **Datasets**: Alpaca, GSM8K, Dolly, TinyStories, HellaSwag
- **File Structure**: `inputs/projects/`, `data/`, `output/`, `app/`

## 💡 DEVELOPMENT GUIDELINES
- Python 3.11 with venv (`./.venv/bin/python3`)
- Test-driven development
- Vector/latent space perspective (not token-based)