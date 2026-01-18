"""Configuration management for Gemini Image MCP server."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Server configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="GEMINI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: str = Field(
        default="",
        description="Google Gemini API key",
    )

    default_model: str = Field(
        default="gemini-2.0-flash-exp",
        description="Default model for image generation",
    )

    default_output_dir: Path | None = Field(
        default=None,
        description="Default directory for saving generated images",
    )

    max_batch_size: int = Field(
        default=10,
        description="Maximum number of images in a batch request",
    )

    default_max_concurrent: int = Field(
        default=3,
        description="Default number of concurrent requests for batch operations",
    )


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset settings (useful for testing)."""
    global _settings
    _settings = None
