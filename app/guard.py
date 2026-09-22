"""Strands hooks: policy enforced in code, and a live audit log of every tool the agent uses.

- Guardrail: no tool that sends, posts, emails or messages may ever run, whatever the model decides.
  Web lookups are allowed only for people you actually met in this debrief.
- Audit: every tool call (and every blocked one) lands in the "What it did on its own" feed.
"""

import json
import re
from typing import Callable

from strands.hooks import AfterToolCallEvent, BeforeToolCallEvent, HookProvider, HookRegistry

OUTBOUND = re.compile(r"send|post|email|message|dm|publish|tweet", re.I)


def _short(v, n=90) -> str:
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)
    return s if len(s) <= n else s[: n - 1] + "…"


class Guardrail(HookProvider):
    def __init__(self, log: Callable[[str, str], None], met_ids: Callable[[], set[str]] | None = None):
        self.log, self.met_ids = log, met_ids

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(BeforeToolCallEvent, self.before)

    def before(self, event: BeforeToolCallEvent) -> None:
        name, args = event.tool_use.get("name", ""), event.tool_use.get("input") or {}
        if OUTBOUND.search(name):
            event.cancel_tool = "Blocked by policy: Room Read never sends anything. Write a draft instead."
            self.log("guardrail", f"Blocked tool '{name}': nothing is ever sent.")
        elif name == "web_lookup" and self.met_ids is not None and args.get("person_id") not in self.met_ids():
            event.cancel_tool = "Blocked by policy: web lookups are only for people I actually met in this debrief."
            self.log("guardrail", f"Blocked a web lookup for someone not met tonight ({args.get('person_id')}).")


class Audit(HookProvider):
    def __init__(self, log: Callable[[str, str], None], agent_name: str):
        self.log, self.agent_name = log, agent_name

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(AfterToolCallEvent, self.after)

    def after(self, event: AfterToolCallEvent) -> None:
        name, args = event.tool_use.get("name", ""), event.tool_use.get("input") or {}
        if event.cancel_message:
            return  # the guardrail already logged it
        status = (event.result or {}).get("status", "?")
        text = " ".join(c.get("text", "") for c in (event.result or {}).get("content", []) if isinstance(c, dict))
        self.log("agent", f"{self.agent_name} → {name}({_short(args, 60)}) {'✓' if status == 'success' else '✗'} {_short(text, 70)}")
