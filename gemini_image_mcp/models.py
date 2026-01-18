"""Pydantic models for input validation."""

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class ModelName(str, Enum):
    """Available Gemini models for image generation."""

    FLASH = "gemini-2.0-flash-exp"
    # Add more models as they become available
    # PRO = "gemini-3-pro-image-preview"


class AspectRatio(str, Enum):
    """Supported aspect ratios for image generation."""

    SQUARE = "1:1"
    LANDSCAPE_16_9 = "16:9"
    PORTRAIT_9_16 = "9:16"
    LANDSCAPE_4_3 = "4:3"
    PORTRAIT_3_4 = "3:4"


class OutputFormat(str, Enum):
    """Output format options."""

    BASE64 = "base64"
    FILE = "file"
    BOTH = "both"


class ImageGenerateInput(BaseModel):
    """Input model for single image generation."""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text description of the image to generate",
    )
    model: ModelName = Field(
        default=ModelName.FLASH,
        description="Model to use for generation",
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.SQUARE,
        description="Aspect ratio of the generated image",
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.BASE64,
        description="Output format: base64, file, or both",
    )
    output_path: Path | None = Field(
        default=None,
        description="File path for output (required if output_format is 'file' or 'both')",
    )

    @field_validator("output_path")
    @classmethod
    def validate_output_path(cls, v: Path | None) -> Path | None:
        """Validate output_path is provided when needed."""
        if v is None:
            return v
        # Ensure parent directory exists or can be created
        parent = v.parent
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        return v


class ImageEditInput(BaseModel):
    """Input model for image editing."""

    image_path: Path = Field(
        ...,
        description="Path to the source image to edit",
    )
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Edit instructions",
    )
    model: ModelName = Field(
        default=ModelName.FLASH,
        description="Model to use for editing",
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.BASE64,
        description="Output format: base64, file, or both",
    )
    output_path: Path | None = Field(
        default=None,
        description="File path for output (required if output_format is 'file' or 'both')",
    )

    @field_validator("image_path")
    @classmethod
    def validate_image_exists(cls, v: Path) -> Path:
        """Validate that the source image exists."""
        if not v.exists():
            raise ValueError(f"Source image not found: {v}")
        if not v.is_file():
            raise ValueError(f"Source path is not a file: {v}")
        return v


class BatchGenerateInput(BaseModel):
    """Input model for batch image generation."""

    prompts: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="List of prompts to generate images for",
    )
    model: ModelName = Field(
        default=ModelName.FLASH,
        description="Model to use for all images",
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.SQUARE,
        description="Aspect ratio for all images",
    )
    max_concurrent: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum number of concurrent requests",
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.BASE64,
        description="Output format: base64, file, or both",
    )
    output_dir: Path | None = Field(
        default=None,
        description="Directory for file output (required if output_format is 'file' or 'both')",
    )

    @field_validator("prompts")
    @classmethod
    def validate_prompts(cls, v: list[str]) -> list[str]:
        """Validate each prompt in the list."""
        for i, prompt in enumerate(v):
            if not prompt.strip():
                raise ValueError(f"Prompt at index {i} is empty")
            if len(prompt) > 5000:
                raise ValueError(f"Prompt at index {i} exceeds 5000 characters")
        return v

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, v: Path | None) -> Path | None:
        """Ensure output directory exists or create it."""
        if v is None:
            return v
        if not v.exists():
            v.mkdir(parents=True, exist_ok=True)
        return v


class ModelInfo(BaseModel):
    """Information about a Gemini model."""

    name: str
    display_name: str
    description: str
    supported_aspect_ratios: list[str]
    supports_editing: bool


class ModelInfoResponse(BaseModel):
    """Response model for get_model_info."""

    models: list[ModelInfo]
    default_model: str
    aspect_ratios: list[str]
