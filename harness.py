# llmharness/harness.py
import litellm
from typing import List, Dict, Any, Optional
from litellm.utils import ModelResponse
import yaml
import logging
from pathlib import Path


logger = logging.getLogger(__name__)

# Define the config directory relative to harness.py
CONFIG_DIR = Path(__file__).parent / "config"

class LLMHarness:
    """Harness for interacting with LLM models through a unified interface"""

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
        self.config = self._load_config(CONFIG_DIR / "models.yaml")
        self.providers = self._load_providers(CONFIG_DIR / "providers.yaml")

    async def complete(
        self,
        model: str,
        prompt: str | List[Dict[str, str]],
        **params: Any
    ) -> ModelResponse:
        """
        Send a completion request to a specific model.

        Args:
            model: Model identifier from config
            prompt: Either a string for one-shot completion or list of messages for chat
            **params: Additional parameters to pass to the model (overrides defaults)

        Returns:
            ModelResponse from litellm

        Raises:
            ValueError: If model or provider configuration is invalid
            Exception: For API call errors
        """
        # Get model config
        model_config = self.config["models"].get(model)
        if not model_config:
            raise ValueError(f"Model {model} not found in config")

        # Get provider
        provider = model_config["provider"]
        if not self.validate_provider(provider):
            raise ValueError(f"Invalid or unconfigured provider: {provider}")

        # Get provider config and API key
        provider_config = self.providers[provider]
        api_key = provider_config.get("api_key")
        if not api_key or api_key.startswith("default_"):
            raise ValueError(f"API key not configured for provider {provider}")

        # Build model string
        format_string = provider_config.get('format', '{model_id}')
        model_identifier = model_config["model_id"]
        full_model_string = format_string.format(model_id=model_identifier)

        # Prepare messages
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        # Build completion params with proper precedence
        completion_params = {
            # Start with default parameters
            **{k: v for k, v in self.DEFAULT_PARAMS.items() if v is not None},
            # Add model-specific parameters from config
            **(model_config.get("parameters", {})),
            # Add core required parameters
            "model": full_model_string,
            "messages": messages,
            "api_key": api_key,
            # User-provided parameters override everything
            **params
        }

        # Make API call
        try:
            response = await litellm.acompletion(**completion_params)
            return response
        except Exception as e:
            logger.error(f"API call error for {model}: {str(e)}")
            raise

    def _load_config(self, config_path: str | Path) -> Dict[str, Any]:
        """Load model configurations from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config if isinstance(config, dict) else {"models": {}}
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
