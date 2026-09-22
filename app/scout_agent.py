"""The scout agent: a Strands agent whose tools are Bright Data's MCP server, running inside the rr-scout Docker sandbox.

Strands connects over stdio through `sbx exec -i`, so the MCP server (and every web request) lives in a microVM whose
network reaches only api.brightdata.com. A hook cleans every Bright Data result before the model reads it: free-text
fields where injections hide are dropped, and injection-looking lines are stripped. The agent has no memory tools. It
returns a typed report, and the host writes that into Cognee with its source and retrieval time.
"""

import json
import os
import re
import tempfile
from datetime import datetime, timezone

from mcp import StdioServerParameters, stdio_client
from pydantic import BaseModel, Field
from strands import Agent
from strands.hooks import AfterToolCallEvent, HookProvider, HookRegistry
from strands.tools.mcp import MCPClient

from app.brain import UNTRUSTED
from app.config import make_model
from app.guard import Audit, Guardrail

SANDBOX = os.getenv("SCOUT_SANDBOX", "rr-scout")
ALLOWED_TOOLS = ["web_data_linkedin_person_profile", "search_engine", "scrape_as_markdown"]
INJECTION = re.compile(r"ignore (all|any|the)? ?(previous|prior|above) instructions|disregard .*instructions|system prompt|"
                       r"you are now|send (me )?(his|her|their|all) (contacts|data)|exfiltrat", re.I)
PROFILE_FIELDS = ("name", "position", "headline", "city", "country_code", "url", "input_url")


def strip_lines(text: str, limit: int) -> tuple[str, int]:
    kept, flagged = [], 0
    for line in text.splitlines():
        if INJECTION.search(line):
            flagged += 1
        else:
            kept.append(line)
    return "\n".join(kept)[:limit], flagged


def clean_result(tool: str, text: str) -> tuple[str, int]:
    """Whitelist what the model may see from one Bright Data tool result."""
    if tool == "web_data_linkedin_person_profile":
        try:
            rec = json.loads(text)
        except json.JSONDecodeError:
            return strip_lines(text, 1500)
        rec = rec[0] if isinstance(rec, list) and rec else rec
        out = {k: rec.get(k) for k in PROFILE_FIELDS if rec.get(k)}
        cc = rec.get("current_company") or {}
        out["current_company"] = cc.get("name") or rec.get("current_company_name")
        out["experience"] = [{"company": e.get("company") or e.get("company_name"), "title": e.get("title"),
                              "start": e.get("start_date"), "end": e.get("end_date")} for e in (rec.get("experience") or [])[:12]]
        out["education"] = [{"school": e.get("title"), "start": e.get("start_year"), "end": e.get("end_year")} for e in (rec.get("education") or [])[:5]]
        flagged = 0
        for k in ("position", "headline"):
            if out.get(k) and INJECTION.search(out[k]):
                out[k], flagged = None, flagged + 1
        out["dropped"] = "about, posts and activity (free text) are never shown to the model"
        return json.dumps(out, ensure_ascii=False), flagged
    if tool == "search_engine":
        links = sorted(set(re.findall(r"https?://[^\s)\"'\]]+", text)))[:15]
        return json.dumps({"links": links}), 0
    return strip_lines(text, 2500)


class Sanitizer(HookProvider):
    """AfterToolCallEvent hook: rewrite Bright Data results before the model reads them."""

    def __init__(self, log):
        self.log = log

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(AfterToolCallEvent, self.after)

    def after(self, event: AfterToolCallEvent) -> None:
        name = event.tool_use.get("name", "")
        if name not in ALLOWED_TOOLS or not event.result:
            return
        raw = " ".join(c.get("text", "") for c in event.result.get("content", []) if isinstance(c, dict))
        cleaned, flagged = clean_result(name, raw)
        event.result = {**event.result, "content": [{"text": cleaned}]}
        if flagged:
            self.log("guardrail", f"Scout: stripped {flagged} injection-looking line(s) from {name} before the model saw it.")


class Job(BaseModel):
    company: str | None = None
    title: str | None = None
    start: str | None = None
    end: str | None = None


class ScoutReport(BaseModel):
    found: bool = Field(description="True only if the tools returned a profile that clearly matches this person")
    name: str | None = None
    linkedin_url: str | None = None
    headline: str | None = None
    current_company: str | None = None
    past_companies: list[Job] = Field(default_factory=list, description="Only employers that appear in tool results")
    education: list[str] = Field(default_factory=list, description="Plain strings like 'University Name (2014–2019)'")
    city: str | None = None
    notes: str = Field(description="What was checked, and what could not be verified")


SYSTEM = f"""You look up ONE person I met, using Bright Data tools that run inside a sandbox.
- If you have their LinkedIn URL, call web_data_linkedin_person_profile with it.
- Otherwise call search_engine with their name, org hint and "linkedin", pick the profile URL that matches, then fetch it.
- Confirm it is the same person (name plus company or context). If unsure, found=false.
- Report only facts present in tool results. If past employers are not in the profile, say so in notes; never guess.
Use at most 4 tool calls. {UNTRUSTED}"""


def _mcp_client(env_file: str) -> MCPClient:
    params = StdioServerParameters(command="sbx", args=[
        "exec", "-i", "--env-file", env_file, "-w", "/home/agent/scout", SANDBOX, "--",
        "env", "PRO_MODE=true", "node", "node_modules/@brightdata/mcp/server.js"])
    return MCPClient(lambda: stdio_client(params), tool_filters={"allowed": ALLOWED_TOOLS}, startup_timeout=60)


def look_up(name: str, linkedin_url: str | None, hint: str, log) -> ScoutReport:
    """Run the scout agent synchronously (call it via asyncio.to_thread)."""
    fd, env_file = tempfile.mkstemp(prefix="rr-scout-", suffix=".env")
    try:
        with os.fdopen(fd, "w") as f:  # 0600 by mkstemp; the token never appears on a command line
            f.write(f"API_TOKEN={os.environ['BRIGHTDATA_API_TOKEN']}\n")
        client = _mcp_client(env_file)
        with client:
            os.unlink(env_file)  # the sandbox has read it; nothing stays on disk
            tools = client.list_tools_sync()
            agent = Agent(model=make_model(1500), tools=tools, callback_handler=None, system_prompt=SYSTEM,
                          # After-hooks run in reverse order: Sanitizer (last) cleans the result before Audit logs it and the model reads it.
                          hooks=[Guardrail(log), Audit(log, "scout agent"), Sanitizer(log)])
            prompt = f"Person: {name}. Org hint: {hint or 'none'}. LinkedIn URL: {linkedin_url or 'unknown'}."
            report = agent(prompt, structured_output_model=ScoutReport).structured_output
    finally:
        if os.path.exists(env_file):
            os.unlink(env_file)
    return report


def retrieved_at() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
