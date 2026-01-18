"""Tests for the Gemini Image MCP server."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gemini_image_mcp.image_utils import (
    bytes_to_base64,
    detect_mime_type,
    generate_filename,
    save_image,
)
from gemini_image_mcp.models import AspectRatio, ModelName, OutputFormat

# Sample PNG data (1x1 transparent pixel)
SAMPLE_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
    b"\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


class TestImageUtils:
    """Tests for image utility functions."""

    def test_bytes_to_base64(self, sample_image_bytes: bytes):
        """Test base64 encoding of image bytes."""
        result = bytes_to_base64(sample_image_bytes)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_save_image(self, tmp_path: Path, sample_image_bytes: bytes):
        """Test saving image bytes to file."""
        output_path = tmp_path / "output.png"
        result = save_image(sample_image_bytes, output_path)

        assert result == output_path
        assert output_path.exists()
        assert output_path.read_bytes() == sample_image_bytes

    def test_save_image_creates_parent_dirs(self, tmp_path: Path, sample_image_bytes: bytes):
        """Test that save_image creates parent directories."""
        output_path = tmp_path / "nested" / "dir" / "output.png"
        result = save_image(sample_image_bytes, output_path)

        assert result == output_path
        assert output_path.exists()

    def test_generate_filename(self):
        """Test filename generation from prompt."""
        filename = generate_filename("A beautiful sunset", index=0)
        assert filename.endswith(".png")
        assert "beautiful" in filename.lower()
        assert "_000" in filename

    def test_generate_filename_without_index(self):
        """Test filename generation without index."""
        filename = generate_filename("Test prompt")
        assert filename.endswith(".png")
        assert "_00" not in filename  # No index suffix

    def test_detect_mime_type_png(self):
        """Test MIME type detection for PNG."""
        assert detect_mime_type(SAMPLE_PNG_BYTES) == "image/png"

    def test_detect_mime_type_jpeg(self):
        """Test MIME type detection for JPEG."""
        jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF"
        assert detect_mime_type(jpeg_bytes) == "image/jpeg"

    def test_detect_mime_type_unknown(self):
        """Test MIME type detection for unknown format."""
        unknown_bytes = b"unknown data"
        assert detect_mime_type(unknown_bytes) == "image/png"  # Default


class TestModels:
    """Tests for Pydantic models."""

    def test_model_name_values(self):
        """Test ModelName enum values."""
        assert ModelName.FLASH.value == "gemini-2.0-flash-exp"

    def test_aspect_ratio_values(self):
        """Test AspectRatio enum values."""
        assert AspectRatio.SQUARE.value == "1:1"
        assert AspectRatio.LANDSCAPE_16_9.value == "16:9"
        assert AspectRatio.PORTRAIT_9_16.value == "9:16"

    def test_output_format_values(self):
        """Test OutputFormat enum values."""
        assert OutputFormat.BASE64.value == "base64"
        assert OutputFormat.FILE.value == "file"
        assert OutputFormat.BOTH.value == "both"


class TestGeminiClient:
    """Tests for the Gemini client."""

    def test_client_requires_api_key(self):
        """Test that client raises error without API key."""
        from gemini_image_mcp.config import reset_settings

        reset_settings()

        with patch.dict("os.environ", {"GEMINI_API_KEY": ""}, clear=True):
            reset_settings()
            from gemini_image_mcp.gemini_client import AuthenticationError, GeminiImageClient

            with pytest.raises(AuthenticationError):
                GeminiImageClient()

    @patch("gemini_image_mcp.gemini_client.genai")
    def test_client_initializes_with_api_key(self, mock_genai):
        """Test that client initializes with API key."""
        from gemini_image_mcp.gemini_client import GeminiImageClient

        GeminiImageClient(api_key="test_key")
        mock_genai.Client.assert_called_once_with(api_key="test_key")


class TestServerTools:
    """Tests for MCP server tools."""

    @pytest.fixture
    def mock_client(self, sample_image_bytes: bytes):
        """Create a mock GeminiImageClient for server tests."""
        mock = MagicMock()
        mock.generate_image = AsyncMock(return_value=sample_image_bytes)
        mock.edit_image = AsyncMock(return_value=sample_image_bytes)
        mock.generate_batch = AsyncMock(return_value=[sample_image_bytes, sample_image_bytes])
        return mock

    @pytest.mark.asyncio
    async def test_generate_image_base64(self, mock_client):
        """Test image generation with base64 output."""
        import gemini_image_mcp.server as server

        server._client = mock_client

        from gemini_image_mcp.server import gemini_image_generate

        result = await gemini_image_generate(
            prompt="A beautiful sunset",
            output_format="base64",
        )

        assert len(result) == 1
        assert result[0].type == "image"
        assert result[0].mimeType == "image/png"

    @pytest.mark.asyncio
    async def test_generate_image_file(self, mock_client, tmp_path: Path):
        """Test image generation with file output."""
        import gemini_image_mcp.server as server

        server._client = mock_client

        from gemini_image_mcp.server import gemini_image_generate

        output_path = tmp_path / "output.png"
        result = await gemini_image_generate(
            prompt="A beautiful sunset",
            output_format="file",
            output_path=str(output_path),
        )

        assert len(result) == 1
        assert result[0].type == "text"
        assert "saved to" in result[0].text.lower()
        assert output_path.exists()

    @pytest.mark.asyncio
    async def test_generate_image_both(self, mock_client, tmp_path: Path):
        """Test image generation with both outputs."""
        import gemini_image_mcp.server as server

        server._client = mock_client

        from gemini_image_mcp.server import gemini_image_generate

        output_path = tmp_path / "output.png"
        result = await gemini_image_generate(
            prompt="A beautiful sunset",
            output_format="both",
            output_path=str(output_path),
        )

        assert len(result) == 2
        # First should be text (file saved)
        assert result[0].type == "text"
        # Second should be image (base64)
        assert result[1].type == "image"

    @pytest.mark.asyncio
    async def test_generate_image_file_requires_path(self, mock_client):
        """Test that file output requires output_path."""
        import gemini_image_mcp.server as server

        server._client = mock_client

        from gemini_image_mcp.server import gemini_image_generate

        result = await gemini_image_generate(
            prompt="A beautiful sunset",
            output_format="file",
        )

        assert len(result) == 1
        assert result[0].type == "text"
        assert "output_path is required" in result[0].text

    @pytest.mark.asyncio
    async def test_edit_image(self, mock_client, temp_image_file: Path):
        """Test image editing."""
        import gemini_image_mcp.server as server

        server._client = mock_client

        from gemini_image_mcp.server import gemini_image_edit

        result = await gemini_image_edit(
            image_path=str(temp_image_file),
            prompt="Add a rainbow",
            output_format="base64",
        )

        assert len(result) == 1
        assert result[0].type == "image"

    @pytest.mark.asyncio
    async def test_edit_image_file_not_found(self, mock_client):
        """Test image editing with non-existent file."""
        import gemini_image_mcp.server as server

        server._client = mock_client

        from gemini_image_mcp.server import gemini_image_edit

        result = await gemini_image_edit(
            image_path="/nonexistent/image.png",
            prompt="Add a rainbow",
        )

        assert len(result) == 1
        assert result[0].type == "text"
        assert "not found" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_batch_generate(self, mock_client):
        """Test batch image generation."""
        import gemini_image_mcp.server as server

        server._client = mock_client

        from gemini_image_mcp.server import gemini_image_batch_generate

        result = await gemini_image_batch_generate(
            prompts=["A cat", "A dog"],
            output_format="base64",
        )

        # Should have summary + 2 images
        assert len(result) == 3
        assert result[0].type == "text"
        assert "2 succeeded" in result[0].text

    @pytest.mark.asyncio
    async def test_batch_generate_empty_prompts(self, mock_client):
        """Test batch generation with empty prompts."""
        import gemini_image_mcp.server as server

        server._client = mock_client

        from gemini_image_mcp.server import gemini_image_batch_generate

        result = await gemini_image_batch_generate(
            prompts=[],
            output_format="base64",
        )

        assert len(result) == 1
        assert result[0].type == "text"
        assert "at least one prompt" in result[0].text.lower()

    def test_get_model_info(self):
        """Test getting model information."""
        from gemini_image_mcp.server import gemini_image_get_model_info

        result = gemini_image_get_model_info()

        assert len(result) == 1
        assert result[0].type == "text"
        assert "gemini" in result[0].text.lower()
        assert "aspect ratio" in result[0].text.lower()
