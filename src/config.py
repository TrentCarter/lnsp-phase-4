try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings  # type: ignore

class Settings(BaseSettings):
    RETRIEVAL_MODE: str = "HYBRID"    # DENSE|GRAPH|HYBRID
    LEXICAL_FALLBACK: bool = True
    MIN_VALID_SCORE: float = 1e-6     # treat <= as degenerate

    class Config:
        env_prefix = "LNSP_"

settings = Settings()