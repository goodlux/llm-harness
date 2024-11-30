from typing import List, Dict, Any, Optional, TypedDict, cast
import litellm  # If you're getting a red squiggle, might need to run: pip install litellm
import yaml
import os
from datetime import datetime
import json
from litellm.utils import ModelResponse, StreamingChoices, Message
import logging
import asyncio

logger = logging.getLogger(__name__)

class ProbeResult(TypedDict):
    """Type definition for probe results"""
    response: str
    model_name: str
    timestamp: str

class LLMHarness:
    """Harness for running probes across multiple LLM models"""

    VALID_PROVIDERS = {
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
        'mistral': 'MISTRAL_API_KEY',  # Changed from mistral-ai to match api_keys.yaml
        'google': 'GOOGLE_API_KEY',
        'together': 'TOGETHER_API_KEY'
    }

    def __init__(self, config_path: str = "config/models.yaml", api_keys_path: str = "config/api_keys.yaml"):
        """Initialize harness with model and API configurations"""
        self.config = self._load_config(config_path)
        self.api_keys = self._load_api_keys(api_keys_path)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load model configurations from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return cast(Dict[str, Any], config) if isinstance(config, dict) else {"models": {}}
        except FileNotFoundError:
            logger.error(f"Model config file not found: {config_path}")
            return {"models": {}}

    def _load_api_keys(self, config_path: str) -> Dict[str, str]:
        """Load API keys from configuration file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return {
                    provider: details['key']
                    for provider, details in config['api_keys'].items()
                }
        except FileNotFoundError:
            logger.error(f"API keys configuration file not found: {config_path}")
            return {}
        except Exception as e:
            logger.error(f"Error loading API keys: {str(e)}")
            return {}

    def validate_provider(self, provider: str) -> bool:
        """Validate provider configuration"""
        if provider not in self.VALID_PROVIDERS:
            logger.error(f"Invalid provider: {provider}")
            logger.info(f"Valid providers are: {list(self.VALID_PROVIDERS.keys())}")
            return False

        if provider not in self.api_keys:
            logger.error(f"No API key found for provider: {provider}")
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

        results: Dict[str, ProbeResult] = {}
        for model_id in models:
            try:
                model_config = self.config["models"][model_id]
                provider = model_config["provider"]
                connection_string = model_config["connection_string"]

                logger.info(f"Attempting {model_id} call:")
                logger.info(f"Provider: {provider}")
                logger.info(f"Connection string: {connection_string}")

                if not self.validate_provider(provider):
                    raise ValueError(f"Invalid or unconfigured provider: {provider}")

                api_key = self.api_keys[provider]
                response = await litellm.acompletion(
                    model=connection_string,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    api_key=api_key
                )

                content = self._extract_content(response)
                model_name = str(getattr(response, 'model', model_id))

                results[model_id] = ProbeResult(
                    response=content,
                    model_name=model_name,
                    timestamp=datetime.now().isoformat()
                )
            except Exception as e:
                logger.error(f"Error with {model_id}: {str(e)}")
                results[model_id] = ProbeResult(
                    response=f"Error: {str(e)}",
                    model_name=model_id,
                    timestamp=datetime.now().isoformat()
                )

        return results
