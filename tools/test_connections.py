# tools/test_connections.py
import yaml
from pathlib import Path
import asyncio
from llm_harness.harness import LLMHarness
import logging
import sys

logging.getLogger("litellm").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open('/dev/null', 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

async def test_connections():
    """Test all configured models through the harness"""
    print(f"\n{Colors.BOLD}Testing Model Connections via Harness:{Colors.END}")
    print("====================================")

    harness = LLMHarness()
    models_config = harness.config["models"]
    results = []
    errors = []

    # Group models by provider for organized output
    provider_models = {}
    for model_name, details in models_config.items():
        if model_name.startswith("#"):
            continue
        provider = details["provider"]
        if provider not in provider_models:
            provider_models[provider] = []
        provider_models[provider].append(model_name)

    # Test each model
    for provider in sorted(provider_models.keys()):
        print(f"\n{Colors.BLUE}Testing {provider} models:{Colors.END}")
        print("─" * 50)

        for model_id in sorted(provider_models[provider]):
            print(f"Testing {model_id}...", end='', flush=True)
            try:
                with SuppressOutput():
                    response = await harness.run_probe(
                        "Hi",
                        models=[model_id]
                    )
                print(f"\r{Colors.GREEN}✓{Colors.END} {model_id}")
                results.append(
                    f"{Colors.GREEN}✅  {model_id:<40}  │  Connected successfully{Colors.END}"
                )
            except Exception as e:
                print(f"\r{Colors.RED}✗{Colors.END} {model_id}")
                error_msg = str(e).split('\n')[0]
                results.append(
                    f"{Colors.RED}❌  {model_id:<40}  │  Error: {error_msg}{Colors.END}"
                )
                errors.append((model_id, str(e)))

    # Display final results
    print(f"\n\n{Colors.BLUE}Connection Test Results:{Colors.END}")
    print(f"{Colors.BOLD}Model{' '*35}  │  Status{Colors.END}")
    print("─" * 80)
    for result in results:
        print(result)

    # Summary
    success_count = sum(1 for r in results if "✅" in r)
    total_count = len(results)
    print(f"\n{Colors.BOLD}Summary:{Colors.END}")
    print(f"Successfully connected to {success_count}/{total_count} models")

    # If there were any errors, offer to show details
    if errors:
        print("\nSome errors occurred. Show detailed error messages? (y/n)", end=' ')
        if input().lower().startswith('y'):
            print("\nDetailed Errors:")
            print("─" * 80)
            for model, error in errors:
                print(f"\n{Colors.BOLD}{model}:{Colors.END}")
                print(f"{error}")

if __name__ == "__main__":
    asyncio.run(test_connections())
