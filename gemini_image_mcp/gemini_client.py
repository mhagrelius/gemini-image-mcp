"""Google Gemini API wrapper for image generation."""

import asyncio
from functools import partial
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from PIL import Image

from gemini_image_mcp.config import get_settings
from gemini_image_mcp.models import AspectRatio, ModelName


class GeminiClientError(Exception):
    """Base exception for Gemini client errors."""

    pass


class AuthenticationError(GeminiClientError):
    """Raised when API key is invalid or missing."""

    pass


class RateLimitError(GeminiClientError):
    """Raised when rate limit is exceeded."""

    retry_after: int | None

    def __init__(self, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class SafetyBlockError(GeminiClientError):
    """Raised when content is blocked by safety filters."""

    pass


class InvalidRequestError(GeminiClientError):
    """Raised for invalid request parameters."""

    pass


class GeminiImageClient:
    """Client for Gemini image generation API."""

    def __init__(self, api_key: str | None = None):
        """Initialize the Gemini client.

        Args:
            api_key: Google Gemini API key. If not provided, uses GEMINI_API_KEY env var.
        """
        settings = get_settings()
        self.api_key = api_key or settings.api_key

        if not self.api_key:
            raise AuthenticationError("Invalid API key. Set GEMINI_API_KEY environment variable.")

        self._client = genai.Client(api_key=self.api_key)

    def _handle_api_error(self, error: Exception) -> None:
        """Convert API errors to appropriate exceptions."""
        error_str = str(error).lower()

        if "401" in error_str or "unauthorized" in error_str or "invalid" in error_str:
            raise AuthenticationError(
                "Invalid API key. Set GEMINI_API_KEY environment variable."
            ) from error

        if "429" in error_str or "rate limit" in error_str or "quota" in error_str:
            # Try to extract retry-after from error
            retry_after = None
            raise RateLimitError(
                "Rate limit exceeded. Retry in a few seconds.",
                retry_after=retry_after,
            ) from error

        if "safety" in error_str or "blocked" in error_str:
            raise SafetyBlockError(
                "Content blocked by safety filters. Try rephrasing your prompt."
            ) from error

        if "400" in error_str or "invalid" in error_str:
            raise InvalidRequestError(f"Invalid request: {error}") from error

        # Re-raise as generic error
        raise GeminiClientError(f"API error: {error}") from error

    def generate_image_sync(
        self,
        prompt: str,
        model: ModelName = ModelName.FLASH,
        aspect_ratio: AspectRatio = AspectRatio.SQUARE,
    ) -> bytes:
        """Generate an image synchronously.

        Args:
            prompt: Text description of the image to generate.
            model: Model to use for generation.
            aspect_ratio: Aspect ratio of the generated image.

        Returns:
            Raw image bytes (PNG format).
        """
        try:
            response = self._client.models.generate_content(
                model=model.value,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_generation_config=types.ImageGenerationConfig(
                        aspect_ratio=aspect_ratio.value,
                    ),
                ),
            )

            # Extract image data from response
            if not response.candidates:
                raise GeminiClientError("No image generated in response")

            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.data:
                    return part.inline_data.data

            raise GeminiClientError("No image data found in response")

        except GeminiClientError:
            raise
        except Exception as e:
            self._handle_api_error(e)
            raise  # Should not reach here

    async def generate_image(
        self,
        prompt: str,
        model: ModelName = ModelName.FLASH,
        aspect_ratio: AspectRatio = AspectRatio.SQUARE,
    ) -> bytes:
        """Generate an image asynchronously.

        Args:
            prompt: Text description of the image to generate.
            model: Model to use for generation.
            aspect_ratio: Aspect ratio of the generated image.

        Returns:
            Raw image bytes (PNG format).
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            partial(
                self.generate_image_sync,
                prompt=prompt,
                model=model,
                aspect_ratio=aspect_ratio,
            ),
        )

    def edit_image_sync(
        self,
        image_path: Path,
        prompt: str,
        model: ModelName = ModelName.FLASH,
    ) -> bytes:
        """Edit an image synchronously with text instructions.

        Args:
            image_path: Path to the source image.
            prompt: Edit instructions.
            model: Model to use for editing.

        Returns:
            Raw image bytes (PNG format).
        """
        try:
            # Load the source image
            image = Image.open(image_path)

            response = self._client.models.generate_content(
                model=model.value,
                contents=[prompt, image],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                ),
            )

            # Extract image data from response
            if not response.candidates:
                raise GeminiClientError("No image generated in response")

            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.data:
                    return part.inline_data.data

            raise GeminiClientError("No image data found in response")

        except GeminiClientError:
            raise
        except Exception as e:
            self._handle_api_error(e)
            raise

    async def edit_image(
        self,
        image_path: Path,
        prompt: str,
        model: ModelName = ModelName.FLASH,
    ) -> bytes:
        """Edit an image asynchronously with text instructions.

        Args:
            image_path: Path to the source image.
            prompt: Edit instructions.
            model: Model to use for editing.

        Returns:
            Raw image bytes (PNG format).
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            partial(
                self.edit_image_sync,
                image_path=image_path,
                prompt=prompt,
                model=model,
            ),
        )

    async def generate_batch(
        self,
        prompts: list[str],
        model: ModelName = ModelName.FLASH,
        aspect_ratio: AspectRatio = AspectRatio.SQUARE,
        max_concurrent: int = 3,
    ) -> list[bytes | Exception]:
        """Generate multiple images in parallel with rate limiting.

        Args:
            prompts: List of prompts to generate images for.
            model: Model to use for all images.
            aspect_ratio: Aspect ratio for all images.
            max_concurrent: Maximum number of concurrent requests.

        Returns:
            List of image bytes or exceptions for failed generations.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def generate_with_semaphore(prompt: str) -> bytes | Exception:
            async with semaphore:
                try:
                    return await self.generate_image(
                        prompt=prompt,
                        model=model,
                        aspect_ratio=aspect_ratio,
                    )
                except Exception as e:
                    return e

        tasks = [generate_with_semaphore(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)


def get_model_info() -> dict[str, Any]:
    """Get information about available models and capabilities."""
    return {
        "models": [
            {
                "name": ModelName.FLASH.value,
                "display_name": "Gemini 2.0 Flash",
                "description": "Fast image generation model",
                "supported_aspect_ratios": [ar.value for ar in AspectRatio],
                "supports_editing": True,
            },
        ],
        "default_model": ModelName.FLASH.value,
        "aspect_ratios": [
            {"value": ar.value, "name": ar.name.replace("_", " ").title()} for ar in AspectRatio
        ],
    }
