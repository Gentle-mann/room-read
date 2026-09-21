"""Host side of the sandboxed scout. Everything that reads raw web text runs inside the rr-scout Docker sandbox."""

import json
import os
import subprocess

from app.config import PRIVATE, ROOT

SANDBOX = os.getenv("SCOUT_SANDBOX", "rr-scout")
SCOUT_DIR = "/home/agent/scout"


def ready() -> bool:
    return bool(os.getenv("BRIGHTDATA_API_TOKEN"))


def install_runner():
    """Copy the runner into the sandbox (idempotent)."""
    subprocess.run(["sbx", "cp", str(ROOT / "sandbox" / "scout.mjs"), f"{SANDBOX}:{SCOUT_DIR}/scout.mjs"], check=True, capture_output=True, timeout=60)


def run(tool: str, args: dict, timeout: int = 240) -> dict:
    """Run one Bright Data tool inside the sandbox; returns whitelisted fields only."""
    if not ready():
        return {"ok": False, "tool": tool, "error": "Bright Data token not set yet"}
    env_file = PRIVATE / ".sandbox.env"  # token reaches the sandbox via a 0600 file, never the command line
    env_file.write_text(f"API_TOKEN={os.environ['BRIGHTDATA_API_TOKEN']}\n")
    os.chmod(env_file, 0o600)
    try:
        proc = subprocess.run(
            ["sbx", "exec", "--env-file", str(env_file), "-w", SCOUT_DIR, SANDBOX, "--", "node", "scout.mjs", tool, json.dumps(args)],
            capture_output=True, text=True, timeout=timeout,
        )
    finally:
        env_file.unlink(missing_ok=True)
    lines = [l for l in proc.stdout.strip().splitlines() if l.startswith("{")]
    if not lines:
        return {"ok": False, "tool": tool, "error": (proc.stderr or proc.stdout)[-300:]}
    return json.loads(lines[-1])


def linkedin_profile(url: str) -> dict:
    return run("web_data_linkedin_person_profile", {"url": url})


def page(url: str) -> dict:
    return run("scrape_as_markdown", {"url": url})


def search(query: str) -> dict:
    return run("search_engine", {"query": query, "engine": "google"})
