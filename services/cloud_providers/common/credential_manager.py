"""
Credential Manager - Securely load API keys from .env
"""

import os
from typing import Optional
from dotenv import load_dotenv


class CredentialManager:
    """
    Manages cloud provider API credentials from .env file

    Usage:
        cm = CredentialManager()
        api_key = cm.get_key("OPENAI_API_KEY")
    """

    def __init__(self, env_file: str = ".env"):
        """
        Initialize credential manager

        Args:
            env_file: Path to .env file (default: ".env")
        """
        # Load .env file (silently fail if not found)
        load_dotenv(env_file, verbose=False, override=False)
        self.env_file = env_file

    def get_key(
        self,
        env_var_name: str,
        required: bool = True,
        provider_name: Optional[str] = None
    ) -> Optional[str]:
        """
        Get API key from environment variable

        Args:
            env_var_name: Name of environment variable (e.g., "OPENAI_API_KEY")
            required: If True, raise ValueError if key not found
            provider_name: Human-readable provider name for error messages

        Returns:
            API key string or None if not required and not found

        Raises:
            ValueError: If required=True and key not found
        """
        api_key = os.getenv(env_var_name)

        if not api_key and required:
            provider = provider_name or env_var_name.replace("_API_KEY", "")
            raise ValueError(
                f"\n{'='*60}\n"
                f"❌ {env_var_name} not found in environment!\n\n"
                f"To use {provider}, please:\n"
                f"1. Create/edit .env file in project root\n"
                f"2. Add your API key:\n"
                f"   {env_var_name}=<your-api-key-here>\n"
                f"3. Restart the service\n\n"
                f"See .env.template for complete example.\n"
                f"{'='*60}"
            )

        return api_key

    def get_model(self, env_var_name: str, default: str) -> str:
        """
        Get model name from environment variable with fallback default

        Args:
            env_var_name: Name of environment variable (e.g., "OPENAI_MODEL_NAME")
            default: Default model name if not found

        Returns:
            Model name string
        """
        return os.getenv(env_var_name, default)

    def check_required_keys(self, required_keys: list[str]) -> dict[str, bool]:
        """
        Check if multiple required keys are present

        Args:
            required_keys: List of environment variable names

        Returns:
            Dict mapping key names to presence boolean
        """
        return {key: bool(os.getenv(key)) for key in required_keys}

    @staticmethod
    def mask_key(api_key: str) -> str:
        """
        Mask API key for safe logging

        Args:
            api_key: Full API key

        Returns:
            Masked key (e.g., "sk-...xyz")
        """
        if not api_key or len(api_key) < 8:
            return "***"

        return f"{api_key[:6]}...{api_key[-3:]}"
