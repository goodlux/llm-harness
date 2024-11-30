# LLM Harness

A lightweight testing harness for working with multiple LLM providers through a unified interface.

## Setup

1. **Install the package**
   ```bash
   pip install -e .
   ```

2. **Configure API Keys**
   ```bash
   # Copy the example config
   cp config/api_keys.example.yaml config/api_keys.yaml

   # Edit config/api_keys.yaml with your API keys
   ```

3. **Configure Models**
   ```bash
   # Copy the example config
   cp config/models.example.yaml config/models.yaml

   # Edit config/models.yaml to specify which models you want to use
   ```

## Validation

1. **Check API Keys**
   ```bash
   python tools/check_api_keys.py
   ```
   This will verify that your API keys are properly configured.

2. **Test Model Connections**
   ```bash
   python tools/test_connections.py
   ```
   This will attempt to connect to each configured model and verify communication.

## Supported Providers

- Anthropic (Claude models)
- OpenAI (GPT models)
- Mistral
- Together.AI
- Google (Gemini models)

## Configuration

### API Keys (config/api_keys.yaml)
```yaml
api_keys:
  anthropic:
    env_var: "ANTHROPIC_API_KEY"
    key: "your_key_here"
  # ... add keys for other providers
```

### Models (config/models.yaml)
```yaml
models:
  "claude-3-opus":
    provider: anthropic
    connection_string: "claude-3-opus-20240229"
    description: "Most capable Claude 3 model"  # description is optional
```

## Project Structure
```
llm-harness/
├── config/                 # Configuration files
│   ├── api_keys.yaml      # Your API keys (not in git)
│   └── models.yaml        # Your model configurations (not in git)
├── src/                   # Source code
│   └── llm_harness/
├── tools/                 # Utility scripts
│   ├── check_api_keys.py
│   └── test_connections.py
└── docs/                  # Documentation
    └── TODO.md           # Future enhancements
```

## Dependencies

The harness uses [liteLLM](https://github.com/BerriAI/litellm) for unified API access to different providers.

## Contributing

See [TODO.md](docs/TODO.md) for planned features and areas needing work.
