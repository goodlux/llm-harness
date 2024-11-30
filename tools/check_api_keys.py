import yaml
import os
from typing import Dict, Any
from rich.console import Console
from rich.table import Table
import yaml
from pathlib import Path

console = Console()

def load_provider_config() -> Dict[str, Any]:
    """Load provider configuration file"""
    try:
        # Always look two directories up, then into config
        config_path = Path(__file__).parent.parent / "config" / "providers.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            return config.get('providers', {})
    except FileNotFoundError as e:
        console.print(f"[red]Provider config not found: {e}")
        return {}

def check_api_keys():
    """Check if API keys are configured properly"""
    providers = load_provider_config()

    table = Table(title="Provider API Key Status")
    table.add_column("Provider", style="cyan")
    table.add_column("Key Info", style="green")
    table.add_column("Status", style="yellow")

    for provider, config in providers.items():
        api_key = config.get('api_key', '')
        if not api_key or api_key.startswith('default_'):
            status = "❌ Not configured"
            key_info = "Missing or default key"
        else:
            status = "✅ Configured"
            # Show first/last few chars of key
            key_info = f"key: {api_key[:3]}...{api_key[-4:]}"

        table.add_row(provider, key_info, status)

    console.print(table)

if __name__ == "__main__":
    console.print("\nChecking Provider API Keys...")
    check_api_keys()
