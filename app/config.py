"""Load config from config.yaml, with env var overrides."""
import os
import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # Env var overrides — .env feeds docker-compose and the app container
    if os.environ.get("VLLM_MODEL"):
        config["generator"]["model"] = os.environ["VLLM_MODEL"]
    if os.environ.get("VLLM_HOST"):
        config["generator"]["vllm_host"] = os.environ["VLLM_HOST"]
    if os.environ.get("VLLM_PORT"):
        config["generator"]["vllm_port"] = int(os.environ["VLLM_PORT"])
    if os.environ.get("DISCORD_TOKEN"):
        config["discord"]["token"] = os.environ["DISCORD_TOKEN"]
        config["discord"]["enabled"] = True

    return config
