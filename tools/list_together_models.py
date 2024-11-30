# tools/list_together_models.py
import requests
from pathlib import Path
import yaml
import json
from typing import Dict, Any

class Colors:
    RED = '\033[91m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    END = '\033[0m'

def load_api_keys() -> Dict[str, Any]:
    """Load API keys from config"""
    config_path = Path(__file__).parent.parent / "config" / "api_keys.yaml"

    if not config_path.exists():
        print(f"{Colors.RED}Error: api_keys.yaml not found at {config_path}{Colors.END}")
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if not config or 'api_keys' not in config:
        print(f"{Colors.RED}Error: 'api_keys' not found in config file{Colors.END}")
        raise KeyError("'api_keys' not found in config")

    if 'together' not in config['api_keys']:
        print(f"{Colors.RED}Error: 'together' key not found in api_keys{Colors.END}")
        raise KeyError("'together' key not found in api_keys")

    return config["api_keys"]

def list_together_models():
    """Get list of available models from Together API"""
    try:
        api_keys = load_api_keys()
        together_key = api_keys['together']['key']

        print(f"{Colors.BLUE}Making request to Together API...{Colors.END}")

        headers = {
            "Authorization": f"Bearer {together_key}",
            "Content-Type": "application/json"
        }

        # Using the correct API endpoint
        response = requests.get(
            "https://api.together.xyz/v1/models",  # Note the /v1/models endpoint
            headers=headers
        )

        print(f"Response status code: {response.status_code}")

        if response.status_code == 200:
            try:
                models = response.json()

                print(f"\n{Colors.GREEN}Available Together.AI Models:{Colors.END}")
                print("=" * 50)

                if isinstance(models, dict) and 'data' in models:
                    models_list = models['data']
                else:
                    models_list = models

                for model in models_list:
                    print(f"\nModel ID: {model.get('id', 'N/A')}")
                    print(f"Name: {model.get('name', 'N/A')}")
                    if 'config' in model:
                        print(f"Context Length: {model['config'].get('context_length', 'N/A')}")
                    print("-" * 30)

                # Save full response for reference
                output_path = Path(__file__).parent.parent / "results" / "together_models.json"
                output_path.parent.mkdir(exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(models, f, indent=2)
                print(f"\nFull results saved to: {output_path}")
            except json.JSONDecodeError as e:
                print(f"{Colors.RED}Error decoding JSON response: {str(e)}{Colors.END}")
                print(f"Raw response: {response.text[:1000]}...")
        else:
            print(f"{Colors.RED}Error: {response.status_code}{Colors.END}")
            print(f"Response text: {response.text}")
    except Exception as e:
        print(f"{Colors.RED}Error: {str(e)}{Colors.END}")

if __name__ == "__main__":
    list_together_models()
