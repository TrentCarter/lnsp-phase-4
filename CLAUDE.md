# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## 🚨 CRITICAL RULES
1. **NEVER USE FUCKING SAMPLE DATA!!!** - NO placeholder/template/hardcoded text lists
2. **NEVER run training without permission** - 
3. **FOCUS on 768D GTR-T5 only** (no STELLA 1024D until requested)
4. **Vec2Text Usage**: See `docs/how_to_use_jxe_and_ielab.md` for CORRECT vec2text implementation

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
- **Vec2Text Usage:** Must use `vec_text_vect_isolated.py` with `--vec2text-backend isolated`


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