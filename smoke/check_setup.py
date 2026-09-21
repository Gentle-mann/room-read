"""Setup check for the hackathon stack. Not product code.

Run from the project root:  .venv/bin/python smoke/check_setup.py
Each check is independent; a summary prints at the end. No key values are printed.
"""

import asyncio
import os
import subprocess
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
# This shell inherits Claude Code's ANTHROPIC_BASE_URL; without this the app's calls go there.
os.environ.pop("ANTHROPIC_BASE_URL", None)

TESLA_JOB = "https://www.tesla.com/careers/search/job/284003"
results = {}


def check(name):
    def wrap(fn):
        t0 = time.time()
        try:
            detail = fn()
            results[name] = f"OK ({time.time() - t0:.1f}s) {detail or ''}".strip()
        except Exception as e:  # noqa: BLE001 - report every failure, keep going
            results[name] = f"FAIL ({time.time() - t0:.1f}s) {type(e).__name__}: {str(e)[:200]}"
        return fn
    return wrap


def has(*names):
    return all(os.getenv(n) for n in names)


@check("keys present")
def _keys():
    problems = []
    if not (has("ANTHROPIC_API_KEY") or has("OPENAI_API_KEY") or has("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")):
        problems.append("agent model: ANTHROPIC_API_KEY, OPENAI_API_KEY, or AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY")
    if not (has("COGNEE_API_URL", "COGNEE_API_KEY") or has("LLM_API_KEY")):
        problems.append("cognee: COGNEE_API_URL + COGNEE_API_KEY (cloud) or LLM_API_KEY (local)")
    if not has("BRIGHTDATA_API_TOKEN"):
        problems.append("BRIGHTDATA_API_TOKEN")
    if problems:
        raise RuntimeError("missing: " + "; ".join(problems))
    agent = "bedrock" if has("AWS_ACCESS_KEY_ID") else "anthropic" if has("ANTHROPIC_API_KEY") else "openai"
    memory = "cognee cloud" if has("COGNEE_API_URL", "COGNEE_API_KEY") else "cognee local"
    return f"agent={agent}, memory={memory}"


@check("strands agent")
def _strands():
    from strands import Agent

    if has("AWS_ACCESS_KEY_ID"):
        from strands.models import BedrockModel

        kwargs = {"region_name": os.getenv("AWS_REGION", "us-west-2"), "max_tokens": 64}
        if os.getenv("BEDROCK_MODEL_ID"):
            kwargs["model_id"] = os.environ["BEDROCK_MODEL_ID"]
        model = BedrockModel(**kwargs)
    elif not has("ANTHROPIC_API_KEY"):
        from strands.models.openai import OpenAIModel

        model = OpenAIModel(
            client_args={"api_key": os.environ["OPENAI_API_KEY"]},
            model_id=os.getenv("OPENAI_MODEL_ID", "gpt-4.1"),
            params={"max_tokens": 64},
        )
    else:
        from strands.models.anthropic import AnthropicModel

        model = AnthropicModel(
            client_args={"api_key": os.environ["ANTHROPIC_API_KEY"]},
            model_id=os.getenv("STRANDS_MODEL_ID", "claude-sonnet-5"),
            max_tokens=64,
        )
    reply = str(Agent(model=model, callback_handler=None)("Reply with exactly: ok")).strip()
    return f"reply={reply[:20]!r}"


@check("cognee remember/recall/forget")
def _cognee():
    import cognee

    cognee.config.system_root_directory(str(ROOT / ".cognee_system"))
    cognee.config.data_root_directory(str(ROOT / ".data_storage"))

    async def run():
        if has("COGNEE_API_URL", "COGNEE_API_KEY"):
            await cognee.serve(url=os.environ["COGNEE_API_URL"], api_key=os.environ["COGNEE_API_KEY"])
        await cognee.remember("Setup check: the hackathon starts at 4 p.m. at Bright Data.", dataset_name="smoke")
        answer = await cognee.recall("What time does the hackathon start?", datasets=["smoke"])
        await cognee.forget(dataset="smoke")
        return answer

    answer = asyncio.run(run())
    return f"recall returned {len(answer)} item(s); first={str(answer[:1])[:120]!r}"


@check("bright data mcp")
def _brightdata():
    from mcp import StdioServerParameters, stdio_client
    from strands.tools.mcp import MCPClient

    env = {"API_TOKEN": os.environ["BRIGHTDATA_API_TOKEN"], "PATH": os.environ["PATH"]}
    client = MCPClient(lambda: stdio_client(StdioServerParameters(command="npx", args=["-y", "@brightdata/mcp"], env=env)))
    with client:
        names = [t.tool_name for t in client.list_tools_sync()]
        res = client.call_tool_sync(tool_use_id="smoke-1", name="scrape_as_markdown", arguments={"url": TESLA_JOB})
    text = " ".join(c.get("text", "") for c in res.get("content", []) if isinstance(c, dict))
    return f"{len(names)} tools; tesla page {'contains People Products' if 'People Products' in text else 'fetched, title not found'} ({len(text)} chars)"


@check("docker sandbox")
def _sbx():
    def sbx(*args, timeout=180):
        return subprocess.run(["sbx", *args], capture_output=True, text=True, timeout=timeout)

    listing = sbx("ls")
    if "sign" in (listing.stderr + listing.stdout).lower() and "in" in listing.stderr.lower():
        raise RuntimeError("not signed in: run `sbx login`")
    if "rr-smoke" not in listing.stdout:
        created = sbx("create", "--name", "rr-smoke", "shell")
        if created.returncode != 0:
            raise RuntimeError(f"create failed: {(created.stderr or created.stdout)[:200]}")
    out = sbx("exec", "rr-smoke", "--", "sh", "-c", "uname -sm; python3 --version 2>&1 || echo no-python")
    if out.returncode != 0:
        raise RuntimeError(f"exec failed: {(out.stderr or out.stdout)[:200]}")
    return out.stdout.strip().replace("\n", " | ")


print("\n=== setup check ===")
for name, status in results.items():
    print(f"{name:32} {status}")
