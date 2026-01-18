"""Pytest fixtures and mocks for tests."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Sample PNG data (1x1 transparent pixel)
SAMPLE_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
    b"\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


@pytest.fixture
def sample_image_bytes():
    """Return sample PNG image bytes."""
    return SAMPLE_PNG_BYTES


@pytest.fixture
def temp_image_file(tmp_path: Path, sample_image_bytes: bytes):
    """Create a temporary image file for testing."""
    image_path = tmp_path / "test_image.png"
    image_path.write_bytes(sample_image_bytes)
    return image_path


@pytest.fixture
def mock_genai_client(sample_image_bytes: bytes):
    """Create a mock Google GenAI client."""
    mock_part = MagicMock()
    mock_part.inline_data.data = sample_image_bytes

    mock_content = MagicMock()
    mock_content.parts = [mock_part]

    mock_candidate = MagicMock()
    mock_candidate.content = mock_content

    mock_response = MagicMock()
    mock_response.candidates = [mock_candidate]

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    return mock_client


@pytest.fixture
def mock_gemini_image_client(mock_genai_client):
    """Create a mock GeminiImageClient."""
    with patch("gemini_image_mcp.gemini_client.genai") as mock_genai:
        mock_genai.Client.return_value = mock_genai_client

        # Import after patching
        from gemini_image_mcp.gemini_client import GeminiImageClient

        # Create client with mock API key
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test_api_key"}):
            from gemini_image_mcp.config import reset_settings

            reset_settings()
            client = GeminiImageClient(api_key="test_api_key")
            yield client


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global state before each test."""
    # Reset settings
    from gemini_image_mcp.config import reset_settings

    reset_settings()

    # Reset server client
    import gemini_image_mcp.server as server

    server._client = None

    yield

    # Cleanup after test
    reset_settings()
    server._client = None
