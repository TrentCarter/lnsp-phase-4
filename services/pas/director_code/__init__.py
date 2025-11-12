"""
Director-Code Service

Port: 6111
LLM: Gemini 2.5 Flash (primary), Claude Sonnet 4.5 (fallback)

Responsibilities:
- Receive job cards from Architect
- Decompose into Manager-level tasks
- Monitor Managers
- Validate acceptance gates
- Report to Architect

Contract: docs/contracts/DIRECTOR_CODE_SYSTEM_PROMPT.md
"""
