import asyncio
from llm_harness.harness import LLMHarness
import yaml
from rich.console import Console
from rich.table import Table
from typing import Dict, Any

console = Console()

async def test_connections():
    """Test connections to all configured models"""
    harness = LLMHarness()

    # Load configurations
    with open("config/models.yaml", 'r') as f:
        models_config = yaml.safe_load(f).get('models', {})

    with open("config/providers.yaml", 'r') as f:
        providers_config = yaml.safe_load(f).get('providers', {})

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
    with open("config/models.yaml", 'r') as f:
        models_config = yaml.safe_load(f).get('models', {})

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
