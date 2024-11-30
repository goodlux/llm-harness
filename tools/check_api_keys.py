# tools/check_api_keys.py
import yaml
from pathlib import Path

# ANSI color codes
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def load_api_keys():
    """Load API keys from config/api_keys.yaml"""
    config_path = Path(__file__).parent.parent / "config" / "api_keys.yaml"

    if not config_path.exists():
        print(f"\n{Colors.RED}❌ Config file not found: {config_path}{Colors.END}")
        print("Please copy config/api_keys.yaml.example to config/api_keys.yaml and add your keys")
        return {}

    with open(config_path) as f:
        return yaml.safe_load(f)["api_keys"]

def check_api_keys():
    """Check for presence of API keys in config"""
    print(f"\n{Colors.BOLD}Checking API Keys in config/api_keys.yaml:{Colors.END}")
    print("=====================================")

    config = load_api_keys()
    found_keys = []
    missing_keys = []
    default_keys = []

    # Calculate maximum lengths for alignment
    max_provider_len = max(len(provider) for provider in config.keys())
    max_env_var_len = max(len(details['env_var']) for details in config.values())

    for provider, details in config.items():
        key = details['key']
        env_var = details['env_var']

        # Format provider name with consistent spacing
        provider_fmt = f"{provider:<{max_provider_len}}"

        if key == f"default_{provider}_key_replace_me":
            default_keys.append(
                f"{Colors.YELLOW}⚠️  {provider_fmt}  │  {env_var:<{max_env_var_len}}  │  (default key, needs replacement){Colors.END}"
            )
        elif key:
            key_preview = f"{key[:4]}...{key[-4:]}"
            found_keys.append(
                f"{Colors.GREEN}✅  {provider_fmt}  │  {env_var:<{max_env_var_len}}  │  key: {key_preview}{Colors.END}"
            )
        else:
            missing_keys.append(
                f"{Colors.RED}❌  {provider_fmt}  │  {env_var:<{max_env_var_len}}  │  (key not found){Colors.END}"
            )

    if found_keys:
        print(f"\n{Colors.BLUE}Valid Keys:{Colors.END}")
        print(f"{Colors.BOLD}Provider      │  Environment Variable     │  Key Info{Colors.END}")
        print("─" * 65)
        for key in found_keys:
            print(key)

    if default_keys:
        print(f"\n{Colors.BLUE}Default Keys (need replacement):{Colors.END}")
        print(f"{Colors.BOLD}Provider      │  Environment Variable     │  Status{Colors.END}")
        print("─" * 65)
        for key in default_keys:
            print(key)

    if missing_keys:
        print(f"\n{Colors.BLUE}Missing Keys:{Colors.END}")
        print(f"{Colors.BOLD}Provider      │  Environment Variable     │  Status{Colors.END}")
        print("─" * 65)
        for key in missing_keys:
            print(key)

if __name__ == "__main__":
    check_api_keys()
