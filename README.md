# Gemini Image MCP Server

An MCP (Model Context Protocol) server for Google Gemini image generation, compatible with Claude Desktop and Claude Code via stdio transport.

## Features

- **Image Generation**: Generate images from text prompts
- **Image Editing**: Edit existing images with text instructions
- **Batch Generation**: Generate multiple images in parallel with rate limiting
- **Model Info**: Query available models and capabilities

## Installation

```bash
# Install the package
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

## Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# Required
GEMINI_API_KEY=your_api_key_here

# Optional
GEMINI_DEFAULT_MODEL=gemini-2.0-flash-exp
GEMINI_DEFAULT_OUTPUT_DIR=./generated_images
GEMINI_MAX_BATCH_SIZE=10
GEMINI_DEFAULT_MAX_CONCURRENT=3
```

### Claude Desktop / Claude Code

Add to your MCP configuration:

```json
{
  "mcpServers": {
    "gemini-image": {
      "command": "gemini-image-mcp",
      "env": {
        "GEMINI_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

Or add via Claude Code CLI:

```bash
claude mcp add gemini-image -- gemini-image-mcp
```

## Tools

### `gemini_image_generate`

Generate a single image from a text prompt.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| prompt | string | required | Text description of the image |
| model | string | gemini-2.0-flash-exp | Model to use |
| aspect_ratio | string | 1:1 | Aspect ratio (1:1, 16:9, 9:16, 4:3, 3:4) |
| output_format | string | base64 | Output format (base64, file, or both) |
| output_path | string | null | File path (required for file output) |

### `gemini_image_edit`

Edit an existing image with text instructions (guided editing, not mask-based).

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| image_path | string | required | Path to source image |
| prompt | string | required | Edit instructions |
| model | string | gemini-2.0-flash-exp | Model to use |
| output_format | string | base64 | Output format |
| output_path | string | null | File path for output |

### `gemini_image_batch_generate`

Generate multiple images in parallel with rate limiting.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| prompts | list[string] | required | List of prompts (max 10) |
| model | string | gemini-2.0-flash-exp | Model for all images |
| aspect_ratio | string | 1:1 | Aspect ratio for all images |
| max_concurrent | int | 3 | Parallel request limit (1-5) |
| output_format | string | base64 | Output format |
| output_dir | string | null | Directory for file output |

### `gemini_image_get_model_info`

Get information about available models, aspect ratios, and capabilities.

**Parameters:** None

## Usage Examples

### In Claude

```
Generate an image of a sunset over mountains
```

```
Edit the image at /path/to/image.png to add a rainbow
```

```
Generate images for these prompts: ["a cat", "a dog", "a bird"]
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run linting
ruff check .
ruff format .

# Run tests
pytest
```

## Error Handling

| Error Type | Description |
|------------|-------------|
| Authentication Error | Invalid or missing API key |
| Rate Limit Error | Too many requests |
| Safety Block | Content blocked by safety filters |
| Invalid Request | Invalid parameters |

## License

MIT
