"""Image processing utilities."""

import base64
from datetime import datetime
from pathlib import Path


def bytes_to_base64(image_bytes: bytes) -> str:
    """Convert image bytes to base64 string.

    Args:
        image_bytes: Raw image data.

    Returns:
        Base64-encoded string.
    """
    return base64.b64encode(image_bytes).decode("utf-8")


def save_image(image_bytes: bytes, output_path: Path) -> Path:
    """Save image bytes to a file.

    Args:
        image_bytes: Raw image data.
        output_path: Path to save the image.

    Returns:
        The path where the image was saved.
    """
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the image data
    output_path.write_bytes(image_bytes)

    return output_path


def generate_filename(prompt: str, index: int | None = None, extension: str = "png") -> str:
    """Generate a filename based on prompt and timestamp.

    Args:
        prompt: The prompt used to generate the image.
        index: Optional index for batch operations.
        extension: File extension (default: png).

    Returns:
        A filename string.
    """
    # Create a safe version of the prompt for filename
    safe_prompt = "".join(c if c.isalnum() or c in " -_" else "" for c in prompt)
    safe_prompt = safe_prompt[:50].strip().replace(" ", "_")

    # Add timestamp for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if index is not None:
        return f"{safe_prompt}_{timestamp}_{index:03d}.{extension}"
    return f"{safe_prompt}_{timestamp}.{extension}"


def get_output_path(
    output_path: Path | None,
    output_dir: Path | None,
    prompt: str,
    index: int | None = None,
) -> Path | None:
    """Determine the output path for an image.

    Args:
        output_path: Explicit output path (takes precedence).
        output_dir: Output directory for generated filename.
        prompt: Prompt used for filename generation.
        index: Optional index for batch operations.

    Returns:
        The resolved output path, or None if no output path is configured.
    """
    if output_path:
        return output_path

    if output_dir:
        filename = generate_filename(prompt, index)
        return output_dir / filename

    return None


def detect_mime_type(image_bytes: bytes) -> str:
    """Detect the MIME type of an image from its bytes.

    Args:
        image_bytes: Raw image data.

    Returns:
        MIME type string (e.g., "image/png", "image/jpeg").
    """
    # Check magic bytes
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    elif image_bytes[:2] == b"\xff\xd8":
        return "image/jpeg"
    elif image_bytes[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    elif image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"

    # Default to PNG as that's what Gemini typically returns
    return "image/png"
