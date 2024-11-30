from typing import List, Dict, Any, Optional, TypedDict, cast
import litellm
import yaml
import os
from datetime import datetime
import json
from litellm.utils import ModelResponse, StreamingChoices, Message

class ProbeResult(TypedDict):
    response: str
    model_name: str
    timestamp: str

async def collect_responses(harness, phrases, tries_dict):
    for phrase in phrases:
        prompt = PROMPT_TEMPLATE.format(input_string=phrase)
        try:
            results = await harness.run_probe(prompt)
            for model, result in results.items():
                if model not in tries_dict:
                    tries_dict[model] = {}
                tries_dict[model][phrase] = result['response'].strip()
        except Exception as e:
            print(f"Error collecting responses for phrase '{phrase}': {e}")
        await asyncio.sleep(2)

def extract_content(response: Any) -> str:
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

class LLMHarness:
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load model configurations from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return cast(Dict[str, Any], config) if isinstance(config, dict) else {"models": {}}
        except FileNotFoundError:
            return {"models": {}}

    def __init__(self, config_path: str = "config/models.yaml"):
        self.config = self._load_config(config_path)
        self.results: Dict[str, ProbeResult] = {}

    async def run_probe(self, prompt: str, models: Optional[List[str]] = None) -> Dict[str, ProbeResult]:
        """Run a semantic probe across specified models"""
        if models is None:
            models = list(self.config["models"].keys())

        results: Dict[str, ProbeResult] = {}
        for model_id in models:
            try:
                model_config = self.config["models"][model_id]
                connection_string = model_config["connection_string"]

                response = await litellm.acompletion(
                    model=connection_string,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )

                # Extract content using the helper function
                content = extract_content(response)

                # Safe extraction of model name
                model_name = str(getattr(response, 'model', model_id))  # Changed model to model_id

                results[model_id] = ProbeResult(  # Changed model to model_id
                    response=content,
                    model_name=model_name,
                    timestamp=datetime.now().isoformat()
                )
            except Exception as e:
                results[model_id] = ProbeResult(  # Changed model to model_id
                    response=f"Error: {str(e)}",
                    model_name=model_id,  # Changed model to model_id
                    timestamp=datetime.now().isoformat()
                )

        return results

    def save_results(self, results: Dict[str, ProbeResult], filename: Optional[str] = None) -> None:
        """Save probe results to JSON file"""
        if filename is None:
            filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        os.makedirs("results", exist_ok=True)
        with open(f"results/{filename}", 'w') as f:
            json.dump(results, f, indent=2)
