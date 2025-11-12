"""
Director-DevSecOps Service

Port: 6114
LLM: Gemini 2.5 Flash (primary), Claude Sonnet 4.5 (fallback)

Responsibilities:
- Receive job cards from Architect
- Decompose into Manager-level CI/CD tasks
- Monitor security scans/SBOM
- Validate gate checks
- Report to Architect

Contract: docs/contracts/DIRECTOR_DEVSECOPS_SYSTEM_PROMPT.md
"""
