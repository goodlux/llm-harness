from typing import List, Dict, Any, Optional, TypedDict, cast
import litellm
import yaml
import os
from datetime import datetime
import json
from litellm.utils import ModelResponse, StreamingChoices, Message
import logging
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)

# Define the config directory relative to harness.py
CONFIG_DIR = Path(__file__).parent.parent.parent / "config"




class ProbeResult(TypedDict):
    """Type definition for probe results"""
    response: str
    model_name: str
    timestamp: str

class LLMHarness:
    """Harness for running probes across multiple LLM models"""

    # Default parameters for all models unless overridden
    DEFAULT_PARAMS = {
        # Core parameters
        "temperature": 0,        # Controls randomness (0-2). Lower is more deterministic
        "max_tokens": None,      # Maximum number of tokens to generate
        "top_p": None,          # Nucleus sampling threshold (0-1)
        "top_k": None,          # Limits vocabulary to top k tokens

        # Advanced generation parameters
        "presence_penalty": None,    # Penalizes repeated tokens (-2 to 2)
        "frequency_penalty": None,   # Penalizes frequent tokens (-2 to 2)
        "repetition_penalty": None,  # Specific penalty for token repetition
        "min_p": None,              # Minimum probability threshold for tokens

        # Context and length control
        "max_retries": None,        # Maximum number of retry attempts
        "context_window": None,     # Maximum context window size
        "timeout": None,            # Request timeout in seconds

        # Response formatting
        "response_format": None,    # Specify response format (e.g., "json")
        "seed": None,               # Random seed for reproducibility
        "tools": None,              # Available tools/functions for the model
        "tool_choice": None,        # Specific tool to use
        "functions": None,          # Available functions (OpenAI format)

        # Stream and processing
        "stream": None,             # Enable streaming responses
        "stop": None,               # Custom stop sequences
        "logit_bias": None,         # Token biasing dictionary

        # Model-specific parameters
        "top_k_return": None,       # Number of responses to return
        "prompt_template": None,    # Custom prompt template
        "roles": None,              # Custom role names for messages

        # Azure specific
        "engine": None,             # Azure deployment name
        "api_version": None,        # API version for Azure

        # Additional controls
        "request_timeout": None,    # Timeout for individual requests
        "validate_response": None,  # Enable response validation
        "num_retries": None,        # Number of retries on failure
    }

    def __init__(self):
        """Initialize harness with model and provider configurations"""
        self.config = self._load_config(CONFIG_DIR / "models.yaml")
        self.providers = self._load_providers(CONFIG_DIR / "providers.yaml")

    def _load_config(self, config_path: str | Path) -> Dict[str, Any]:
        """Load model configurations from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return cast(Dict[str, Any], config) if isinstance(config, dict) else {"models": {}}
        except FileNotFoundError:
            logger.error(f"Model config file not found: {config_path}")
            return {"models": {}}

    def _load_providers(self, config_path: str | Path) -> Dict[str, Any]:
        """Load provider configurations including API keys"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('providers', {})
        except FileNotFoundError:
            logger.error(f"Provider config not found: {config_path}")
            return {}

    def validate_provider(self, provider: str) -> bool:
        """Validate provider configuration"""
        if provider not in self.providers:
            logger.error(f"Invalid provider: {provider}")
            logger.info(f"Valid providers are: {list(self.providers.keys())}")
            return False
        return True

    @staticmethod
    def _extract_content(response: Any) -> str:
        """Safely extract content from various response types"""
        try:
            # Handle litellm response
            if isinstance(response, ModelResponse):
                if not response.choices or len(response.choices) == 0:
                    return ""

                choice = response.choices[0]
                # Handle different choice types
                if isinstance(choice, dict):
                    message_dict = cast(Dict[str, Any], choice.get('message', {}))
                    return str(message_dict.get('content', ''))

                # Handle streaming response
                if hasattr(choice, 'delta'):
                    delta = getattr(choice, 'delta')
                    return str(getattr(delta, 'content', ''))

                # Handle standard response
                if hasattr(choice, 'message'):
                    msg = getattr(choice, 'message')
                    return str(getattr(msg, 'content', ''))

                return str(choice)

            # Handle dict-like responses
            if isinstance(response, dict):
                if 'choices' in response:
                    choice = response['choices'][0]
                    if isinstance(choice, dict) and 'message' in choice:
                        return str(choice['message'].get('content', ''))

            # Handle object-style responses
            if hasattr(response, 'choices'):
                choices = getattr(response, 'choices')
                if choices and len(choices) > 0:
                    choice = choices[0]
                    if hasattr(choice, 'message'):
                        msg = getattr(choice, 'message')
                        if hasattr(msg, 'content'):
                            return str(msg.content)

            # Fallback: try to convert response to string
            return str(response)
        except Exception as e:
            return f"Error extracting content: {str(e)}"

    async def run_probe(self, prompt: str, models: Optional[List[str]] = None) -> Dict[str, ProbeResult]:
        """Run a semantic probe across specified models"""
        if models is None:
            models = list(self.config["models"].keys())

        logger.info("Starting run_probe")
        logger.info(f"Models to test: {models}")

        results: Dict[str, ProbeResult] = {}
        for model_id in models:
            try:
                logger.info(f"\nProcessing model: {model_id}")

                # Get model config
                model_config = self.config["models"].get(model_id)
                if not model_config:
                    raise ValueError(f"Model {model_id} not found in config")
                logger.info(f"Model config: {model_config}")

                # Get provider
                provider = model_config["provider"]
                if not self.validate_provider(provider):
                    raise ValueError(f"Invalid or unconfigured provider: {provider}")
                logger.info(f"Using provider: {provider}")

                # Get provider config
                provider_config = self.providers[provider]
                api_key = provider_config.get("api_key")
                if not api_key or api_key.startswith("default_"):
                    raise ValueError(f"API key not configured for provider {provider}")

                # Build model string
                format_string = provider_config.get('format', '{model_id}')
                model_identifier = model_config["model_id"]
                full_model_string = format_string.format(model_id=model_identifier)
                logger.info(f"Full model string: {full_model_string}")

                # Build completion params
                completion_params = {
                    "model": full_model_string,
                    "messages": [{"role": "user", "content": prompt}],
                    "api_key": api_key
                }
                logger.info(f"Completion params (excluding api_key): {dict(model=completion_params['model'])}")

                # Make API call
                try:
                    response = await litellm.acompletion(**completion_params)
                    logger.info("Got response from litellm")
                    content = self._extract_content(response)
                    model_name = str(getattr(response, 'model', model_id))

                    results[model_id] = ProbeResult(
                        response=content,
                        model_name=model_name,
                        timestamp=datetime.now().isoformat()
                    )
                except Exception as api_error:
                    logger.error(f"API call error for {model_id}: {str(api_error)}")
                    raise ValueError(f"API call failed: {str(api_error)}")

            except Exception as e:
                logger.error(f"Error with {model_id}: {str(e)}")
                results[model_id] = ProbeResult(
                    response=f"Error: {str(e)}",
                    model_name=model_id,
                    timestamp=datetime.now().isoformat()
                )

        return results
