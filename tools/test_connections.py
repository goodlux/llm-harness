import asyncio
from llmharness import LLMHarness
import yaml
from rich.console import Console
from rich.table import Table
from typing import Dict, Any
from pathlib import Path

console = Console()

def load_config(filename: str) -> Dict[str, Any]:
    """Load configuration file using relative path"""
    try:
        # Adjust path to look in parent directory's config folder
        config_path = Path(__file__).parent.parent / "config" / filename
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError as e:
        console.print(f"[red]Error loading {filename}: {str(e)}")
        return {}

async def test_connections():
    """Test connections to all configured models"""
    harness = LLMHarness()

    # Load configurations
    models_config = load_config('models.yaml').get('models', {})
    providers_config = load_config('providers.yaml').get('providers', {})

    # Create results table
    table = Table(title="Model Connection Test Results")
    table.add_column("Model", style="cyan")
    table.add_column("Provider", style="yellow")
    table.add_column("Model ID", style="blue")
    table.add_column("Status", style="green")

    # Test each model
    test_prompt = "Respond with 'OK' if you can read this."

    for model_id, model_config in models_config.items():
        provider = model_config['provider']
        model_identifier = model_config.get('model_id', 'Unknown')
        try:
            console.print(f"\nTesting {model_id}...")
            results = await harness.run_probe(test_prompt, models=[model_id])

            if model_id in results:
                response = results[model_id]['response']
                if "Error" in response:
                    status = f"❌ {response}"
                else:
                    status = "✅ Connected"
            else:
                status = "❌ No response"

        except Exception as e:
            status = f"❌ Error: {str(e)}"

        table.add_row(
            model_id,
            provider,
            model_identifier,
            status
        )

    console.print("\nConnection Test Results:")
    console.print(table)

def group_models_by_provider(models_config: Dict) -> Dict[str, list]:
    """Group models by provider for organized testing"""
    grouped = {}
    for model_id, config in models_config.items():
        provider = config['provider']
        if provider not in grouped:
            grouped[provider] = []
        grouped[provider].append(model_id)
    return grouped

async def main():
    """Main test routine with organized output"""
    console.print("\nTesting Model Connections via Harness:")
    console.print("====================================\n")

    # Load model configurations
    models_config = load_config('models.yaml').get('models', {})

    # Group models by provider
    grouped_models = group_models_by_provider(models_config)

    # Print testing plan
    for provider, models in grouped_models.items():
        console.print(f"\nTesting {provider} models:")
        console.print("──────────────────────────────────────────────────")
        for model in models:
            console.print(f"Testing {model}...")

    # Run actual tests
    await test_connections()

if __name__ == "__main__":
    asyncio.run(main())
