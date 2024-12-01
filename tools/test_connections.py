import asyncio
import yaml
from pathlib import Path
from harness import LLMHarness

harness = LLMHarness()

# Debug print at the start
config_file = Path("config/models.yaml").resolve()
print(f"Looking for config at: {config_file}")

async def test_models():
    with open("config/models.yaml") as f:
        models = yaml.safe_load(f)['models']

    for model_id, config in models.items():
        print(f"Testing {model_id}...")
        try:
            result = await harness.run_probe("test", models=[model_id])
            print("✅" if model_id in result else "❌")
        except Exception as e:
            print(f"❌ {e}")

if __name__ == "__main__":
    asyncio.run(test_models())
