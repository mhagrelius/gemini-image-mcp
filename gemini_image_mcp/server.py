"""FastMCP server for Gemini image generation."""

from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

from gemini_image_mcp.config import get_settings
from gemini_image_mcp.gemini_client import (
    AuthenticationError,
    GeminiClientError,
    GeminiImageClient,
    InvalidRequestError,
    RateLimitError,
    SafetyBlockError,
    get_model_info,
)
from gemini_image_mcp.image_utils import (
    bytes_to_base64,
    detect_mime_type,
    get_output_path,
    save_image,
)
from gemini_image_mcp.models import AspectRatio, ModelName, OutputFormat

# Initialize FastMCP server
mcp = FastMCP(name="gemini-image")

# Lazy-initialized client
_client: GeminiImageClient | None = None


def get_client() -> GeminiImageClient:
    """Get or create the Gemini client."""
    global _client
    if _client is None:
        _client = GeminiImageClient()
    return _client


def format_error(error: Exception) -> str:
    """Format an error message for the user."""
    if isinstance(error, AuthenticationError):
        return f"Authentication error: {error}"
    elif isinstance(error, RateLimitError):
        msg = f"Rate limit error: {error}"
        if error.retry_after:
            msg += f" Retry after {error.retry_after} seconds."
        return msg
    elif isinstance(error, SafetyBlockError):
        return f"Safety error: {error}"
    elif isinstance(error, InvalidRequestError):
        return f"Invalid request: {error}"
    elif isinstance(error, GeminiClientError):
        return f"API error: {error}"
    else:
        return f"Unexpected error: {error}"


@mcp.tool()
async def gemini_image_generate(
    prompt: str,
    model: str = "gemini-2.0-flash-exp",
    aspect_ratio: str = "1:1",
    output_format: str = "base64",
    output_path: str | None = None,
) -> list[TextContent | ImageContent]:
    """Generate an image from a text prompt using Google Gemini.

    Args:
        prompt: Text description of the image to generate.
        model: Model to use (default: gemini-2.0-flash-exp).
        aspect_ratio: Aspect ratio (1:1, 16:9, 9:16, 4:3, 3:4).
        output_format: Output format (base64, file, or both).
        output_path: File path for output (required if output_format is 'file' or 'both').

    Returns:
        Generated image as base64 and/or file path.
    """
    try:
        # Validate enum values
        model_enum = ModelName(model)
        aspect_ratio_enum = AspectRatio(aspect_ratio)
        output_format_enum = OutputFormat(output_format)

        # Validate output_path requirement
        path = Path(output_path) if output_path else None
        if output_format_enum in (OutputFormat.FILE, OutputFormat.BOTH) and not path:
            return [
                TextContent(
                    type="text",
                    text="Error: output_path is required when output_format is 'file' or 'both'",
                )
            ]

        # Generate the image
        client = get_client()
        image_bytes = await client.generate_image(
            prompt=prompt,
            model=model_enum,
            aspect_ratio=aspect_ratio_enum,
        )

        results: list[TextContent | ImageContent] = []
        mime_type = detect_mime_type(image_bytes)

        # Handle file output
        if output_format_enum in (OutputFormat.FILE, OutputFormat.BOTH) and path:
            saved_path = save_image(image_bytes, path)
            results.append(TextContent(type="text", text=f"Image saved to: {saved_path}"))

        # Handle base64 output
        if output_format_enum in (OutputFormat.BASE64, OutputFormat.BOTH):
            base64_data = bytes_to_base64(image_bytes)
            results.append(ImageContent(type="image", data=base64_data, mimeType=mime_type))

        return results

    except ValueError as e:
        return [TextContent(type="text", text=f"Validation error: {e}")]
    except GeminiClientError as e:
        return [TextContent(type="text", text=format_error(e))]
    except Exception as e:
        return [TextContent(type="text", text=f"Unexpected error: {e}")]


@mcp.tool()
async def gemini_image_edit(
    image_path: str,
    prompt: str,
    model: str = "gemini-2.0-flash-exp",
    output_format: str = "base64",
    output_path: str | None = None,
) -> list[TextContent | ImageContent]:
    """Edit an existing image using text instructions.

    This performs guided editing via text prompt, not mask-based inpainting.

    Args:
        image_path: Path to the source image to edit.
        prompt: Edit instructions describing the desired changes.
        model: Model to use (default: gemini-2.0-flash-exp).
        output_format: Output format (base64, file, or both).
        output_path: File path for output (required if output_format is 'file' or 'both').

    Returns:
        Edited image as base64 and/or file path.
    """
    try:
        # Validate enum values
        model_enum = ModelName(model)
        output_format_enum = OutputFormat(output_format)

        # Validate paths
        source_path = Path(image_path)
        if not source_path.exists():
            return [TextContent(type="text", text=f"Error: Source image not found: {image_path}")]

        dest_path = Path(output_path) if output_path else None
        if output_format_enum in (OutputFormat.FILE, OutputFormat.BOTH) and not dest_path:
            return [
                TextContent(
                    type="text",
                    text="Error: output_path is required when output_format is 'file' or 'both'",
                )
            ]

        # Edit the image
        client = get_client()
        image_bytes = await client.edit_image(
            image_path=source_path,
            prompt=prompt,
            model=model_enum,
        )

        results: list[TextContent | ImageContent] = []
        mime_type = detect_mime_type(image_bytes)

        # Handle file output
        if output_format_enum in (OutputFormat.FILE, OutputFormat.BOTH) and dest_path:
            saved_path = save_image(image_bytes, dest_path)
            results.append(TextContent(type="text", text=f"Image saved to: {saved_path}"))

        # Handle base64 output
        if output_format_enum in (OutputFormat.BASE64, OutputFormat.BOTH):
            base64_data = bytes_to_base64(image_bytes)
            results.append(ImageContent(type="image", data=base64_data, mimeType=mime_type))

        return results

    except ValueError as e:
        return [TextContent(type="text", text=f"Validation error: {e}")]
    except GeminiClientError as e:
        return [TextContent(type="text", text=format_error(e))]
    except Exception as e:
        return [TextContent(type="text", text=f"Unexpected error: {e}")]


@mcp.tool()
async def gemini_image_batch_generate(
    prompts: list[str],
    model: str = "gemini-2.0-flash-exp",
    aspect_ratio: str = "1:1",
    max_concurrent: int = 3,
    output_format: str = "base64",
    output_dir: str | None = None,
) -> list[TextContent | ImageContent]:
    """Generate multiple images in parallel with rate limiting.

    Args:
        prompts: List of prompts to generate images for (max 10).
        model: Model to use for all images (default: gemini-2.0-flash-exp).
        aspect_ratio: Aspect ratio for all images (1:1, 16:9, 9:16, 4:3, 3:4).
        max_concurrent: Maximum number of concurrent requests (1-5, default: 3).
        output_format: Output format (base64, file, or both).
        output_dir: Directory for file output (required if output_format is 'file' or 'both').

    Returns:
        List of generated images as base64 and/or file paths.
    """
    try:
        # Validate inputs
        settings = get_settings()
        if len(prompts) > settings.max_batch_size:
            return [
                TextContent(
                    type="text",
                    text=f"Error: Maximum {settings.max_batch_size} prompts allowed per batch",
                )
            ]

        if not prompts:
            return [TextContent(type="text", text="Error: At least one prompt is required")]

        # Validate enum values
        model_enum = ModelName(model)
        aspect_ratio_enum = AspectRatio(aspect_ratio)
        output_format_enum = OutputFormat(output_format)

        # Validate output_dir requirement
        dir_path = Path(output_dir) if output_dir else None
        if output_format_enum in (OutputFormat.FILE, OutputFormat.BOTH) and not dir_path:
            return [
                TextContent(
                    type="text",
                    text="Error: output_dir is required when output_format is 'file' or 'both'",
                )
            ]

        # Ensure output directory exists
        if dir_path:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Generate images in parallel
        client = get_client()
        batch_results = await client.generate_batch(
            prompts=prompts,
            model=model_enum,
            aspect_ratio=aspect_ratio_enum,
            max_concurrent=min(max_concurrent, 5),
        )

        results: list[TextContent | ImageContent] = []
        success_count = 0
        error_count = 0

        for i, (prompt, result) in enumerate(zip(prompts, batch_results, strict=True)):
            if isinstance(result, Exception):
                error_count += 1
                results.append(
                    TextContent(
                        type="text",
                        text=f"[{i + 1}] Error for prompt '{prompt[:50]}...': {format_error(result)}",
                    )
                )
            else:
                success_count += 1
                image_bytes = result
                mime_type = detect_mime_type(image_bytes)

                # Handle file output
                if output_format_enum in (OutputFormat.FILE, OutputFormat.BOTH) and dir_path:
                    file_path = get_output_path(None, dir_path, prompt, i)
                    if file_path:
                        saved_path = save_image(image_bytes, file_path)
                        results.append(
                            TextContent(
                                type="text",
                                text=f"[{i + 1}] Image saved to: {saved_path}",
                            )
                        )

                # Handle base64 output
                if output_format_enum in (OutputFormat.BASE64, OutputFormat.BOTH):
                    base64_data = bytes_to_base64(image_bytes)
                    results.append(ImageContent(type="image", data=base64_data, mimeType=mime_type))

        # Add summary
        results.insert(
            0,
            TextContent(
                type="text",
                text=f"Batch complete: {success_count} succeeded, {error_count} failed",
            ),
        )

        return results

    except ValueError as e:
        return [TextContent(type="text", text=f"Validation error: {e}")]
    except GeminiClientError as e:
        return [TextContent(type="text", text=format_error(e))]
    except Exception as e:
        return [TextContent(type="text", text=f"Unexpected error: {e}")]


@mcp.tool()
def gemini_image_get_model_info() -> list[TextContent]:
    """Get information about available Gemini models and their capabilities.

    Returns:
        Information about available models, supported aspect ratios, and capabilities.
    """
    info = get_model_info()

    lines = ["# Gemini Image Generation Models\n"]

    for model in info["models"]:
        lines.append(f"## {model['display_name']}")
        lines.append(f"- **Model ID:** `{model['name']}`")
        lines.append(f"- **Description:** {model['description']}")
        lines.append(f"- **Supports Editing:** {'Yes' if model['supports_editing'] else 'No'}")
        lines.append(f"- **Aspect Ratios:** {', '.join(model['supported_aspect_ratios'])}")
        lines.append("")

    lines.append(f"**Default Model:** `{info['default_model']}`")
    lines.append("\n## Available Aspect Ratios")
    for ar in info["aspect_ratios"]:
        lines.append(f"- `{ar['value']}` ({ar['name']})")

    return [TextContent(type="text", text="\n".join(lines))]


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
