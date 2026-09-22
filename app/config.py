"""Settings, model selection and Cognee startup. Import this module before cognee anywhere."""

import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
# The dev shell inherits Claude Code's ANTHROPIC_BASE_URL; the app must talk to the real API.
os.environ.pop("ANTHROPIC_BASE_URL", None)
# Blank lines in .env (e.g. "AWS_ACCESS_KEY_ID=") must count as unset: Cognee treats "" as real AWS keys
# and switches its file layer to S3.
for _k in [k for k, v in os.environ.items() if v == ""]:
    del os.environ[_k]

TZ = ZoneInfo(os.getenv("TZ_NAME", "America/Los_Angeles"))
PRIVATE = ROOT / "data" / "private"
PRIVATE.mkdir(parents=True, exist_ok=True)
FEEDS = ROOT / "data" / "feeds"

EVENT_ID = os.getenv("EVENT_ID", "2026-09-21")
EVENT_NAME = os.getenv("EVENT_NAME", "Battle of the Personal Brains")
ROOM_DATASET = f"room-{EVENT_ID}"
PEOPLE_DATASET = "people"
ME_DATASET = "me"

# Temporary guest-list export (session scratch folder). Never copied into the project.
GUESTS_DIR = Path(os.getenv("GUESTS_DIR", "")).expanduser() if os.getenv("GUESTS_DIR") else None
HOSTS = [h.strip() for h in os.getenv("EVENT_HOSTS", "Vasilije Markovic,Dave Nielsen").split(",") if h.strip()]


def has(*names: str) -> bool:
    return all(os.getenv(n) for n in names)


def now() -> datetime:
    return datetime.now(TZ)


def make_model(max_tokens: int = 2000):
    """Bedrock if AWS credentials exist, else Anthropic, else OpenAI."""
    if has("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        from strands.models import BedrockModel

        kwargs = {"region_name": os.getenv("AWS_REGION", "us-west-2"), "max_tokens": max_tokens}
        if os.getenv("BEDROCK_MODEL_ID"):
            kwargs["model_id"] = os.environ["BEDROCK_MODEL_ID"]
        return BedrockModel(**kwargs)
    if has("ANTHROPIC_API_KEY"):
        from strands.models.anthropic import AnthropicModel

        return AnthropicModel(
            client_args={"api_key": os.environ["ANTHROPIC_API_KEY"]},
            model_id=os.getenv("STRANDS_MODEL_ID", "claude-sonnet-5"),
            max_tokens=max_tokens,
        )
    from strands.models.openai import OpenAIModel

    return OpenAIModel(
        client_args={"api_key": os.environ["OPENAI_API_KEY"]},
        model_id=os.getenv("OPENAI_MODEL_ID", "gpt-4.1"),
        params={"max_tokens": max_tokens},
    )


_cognee_ready = False


async def init_cognee():
    """Point Cognee at project-local storage, or at Cognee Cloud when configured."""
    global _cognee_ready
    import cognee

    if _cognee_ready:
        return cognee
    cognee.config.system_root_directory(str(ROOT / ".cognee_system"))
    cognee.config.data_root_directory(str(ROOT / ".data_storage"))
    if has("COGNEE_API_URL", "COGNEE_API_KEY"):
        await cognee.serve(url=os.environ["COGNEE_API_URL"], api_key=os.environ["COGNEE_API_KEY"])
    _cognee_ready = True
    return cognee
