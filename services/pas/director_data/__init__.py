"""
Director-Data Service

Port: 6113
LLM: Claude Sonnet 4.5 (primary), Gemini 2.5 Pro (fallback)

Responsibilities:
- Receive job cards from Architect
- Decompose into Manager-level data tasks
- Monitor ingestion/validation
- Validate schema gates
- Report to Architect

Contract: docs/contracts/DIRECTOR_DATA_SYSTEM_PROMPT.md
"""
