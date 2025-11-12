"""
Director-Docs Service

Port: 6115
LLM: Claude Sonnet 4.5 (primary), Gemini 2.5 Pro (fallback)

Responsibilities:
- Receive job cards from Architect
- Decompose into Manager-level documentation tasks
- Monitor doc generation/review
- Validate completeness gates
- Report to Architect

Contract: docs/contracts/DIRECTOR_DOCS_SYSTEM_PROMPT.md
"""
