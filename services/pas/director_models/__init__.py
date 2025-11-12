"""
Director-Models Service

Port: 6112
LLM: Claude Sonnet 4.5 (primary), Gemini 2.5 Pro (fallback)

Responsibilities:
- Receive job cards from Architect
- Decompose into Manager-level training tasks
- Monitor training/evaluation
- Validate KPI gates
- Report to Architect

Contract: docs/contracts/DIRECTOR_MODELS_SYSTEM_PROMPT.md
"""
