"""The debrief agent: one Strands agent turns a spoken note into memory by choosing tools.

The exact rules (name matching, promise timers, the sandboxed scout, memory writes) are tools; the model decides
which to use and when. It pauses with a Strands interrupt when a name is ambiguous, and hooks enforce policy and
log every step.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Literal

from pydantic import BaseModel
from strands import Agent, ToolContext, tool

from app import memory
from app.brain import STYLE, UNTRUSTED, Card
from app.config import EVENT_ID, EVENT_NAME, ME_DATASET, PEOPLE_DATASET, ROOM_DATASET, make_model, now
from app.guard import Audit, Guardrail
from app.guests import resolve


class PersonOut(BaseModel):
    person_id: str
    card: Card


class DebriefOutcome(BaseModel):
    summary: str
    people: list[PersonOut]


@dataclass
class DebriefRun:
    text: str
    ledger: object
    guests: Callable[[], list]
    log: Callable[[str, str], None]
    background: Callable[..., None]  # background(kind, **kwargs) schedules memory writes and web lookups
    at: datetime = field(default_factory=now)
    met: dict = field(default_factory=dict)  # person_id -> {"name", "guest"}
    interactions: dict = field(default_factory=dict)  # person_id -> interaction id


SYSTEM = """You are Room Read, my networking memory. I just recorded a short spoken note after meeting people at {event}.
The current time is {now}. You act only through tools; nothing you do is ever sent to anyone.

For every person I mention:
1. resolve_person first. If several guests share the name it pauses and asks me; wait for that.
2. recall_memory(scope="people") to check whether I have met them before, and what we said.
3. remember_meeting with only what I actually said.
4. set_promise_timer ONLY for commitments I explicitly stated ("I promised…", "I'll send…", "we agreed to…").
   Never create a promise from your own suggestion or from the follow-up draft. Resolve relative times; a day with no hour means 09:00.
5. web_lookup for them, which runs in the background, in a sandbox.
Call recall_memory(scope="me") once to learn my ranked goals, offers and stories.

Then return a card per person: reasons ranked by MY goals, each with an honest source label
("tonight" = what I said, "memory" = earlier meetings, "their Luma profile", "my notes"), and a short follow-up draft in my voice that makes no new commitments.
Never invent facts. {style} {untrusted}"""


def build_agent(run: DebriefRun) -> Agent:
    @tool(context=True)
    def resolve_person(name: str, org_hint: str = "", tool_context: ToolContext = None) -> str:
        """Match a name I said to tonight's guest list and open or reuse their record. Call first for each person.
        If several guests match, this pauses and asks me which one I meant."""
        r = resolve(name, run.guests())
        g = r.matches[0] if r.status == "unique" else None
        if r.status == "ambiguous":
            answer = tool_context.interrupt("which_person", reason={"mention": name, "candidates": [c.public() for c in r.matches]})
            g = next((c for c in r.matches if c.id == answer), None)
        pid = run.ledger.upsert_person(
            name=g.name if g else name, guest_id=g.id if g else None, links=g.links if g else {}, bio=g.bio if g else "",
            source="debrief" if g else "debrief (not on the guest list)", met_at=run.at.isoformat(), event_id=EVENT_ID,
        )
        run.met[pid] = {"name": g.name if g else name, "guest": g, "org_hint": org_hint}
        return json.dumps({"person_id": pid, "name": run.met[pid]["name"], "on_guest_list": bool(g),
                           "luma_bio": g.bio if g else None, "links": g.links if g else {}})

    @tool
    async def recall_memory(question: str, scope: Literal["me", "people", "room"]) -> str:
        """Ask my memory. scope="me": my goals, offers, stories. "people": everyone I have met before. "room": tonight's guest list."""
        dataset = {"me": ME_DATASET, "people": PEOPLE_DATASET, "room": ROOM_DATASET}[scope]
        return (await memory.recall(question, [dataset], context_only=scope == "me"))[:3000] or "(nothing yet)"

    @tool
    def remember_meeting(person_id: str, what_i_heard: str, their_needs: list[str], i_promised: list[str]) -> str:
        """Save tonight's meeting with this person to permanent memory. Only first-hand facts from my note."""
        if person_id not in run.met:
            return "unknown person_id: call resolve_person first"
        name, g = run.met[person_id]["name"], run.met[person_id]["guest"]
        run.interactions[person_id] = run.ledger.add_interaction(
            person_id=person_id, at=run.at.isoformat(), event_id=EVENT_ID, raw_text=run.text, summary=what_i_heard)
        docs = ([g.as_doc(EVENT_NAME)] if g else []) + [
            f"On {run.at:%Y-%m-%d at %H:%M} at {EVENT_NAME}, I met {name}. What I heard: {what_i_heard} "
            f"They are looking for: {'; '.join(their_needs) or 'n/a'}. I promised: {'; '.join(i_promised) or 'nothing'}."]
        run.background("remember", pid=person_id, docs=docs, label=name)
        return "saved; permanent memory is updating in the background"

    @tool
    def set_promise_timer(person_id: str, promise: str, due_iso: str | None = None) -> str:
        """Start a timer for something I committed to do for this person. due_iso is ISO 8601 with offset."""
        due = due_iso or (run.at + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0).isoformat()
        run.ledger.add_promise(person_id=person_id, text=promise, due_at=due, created_at=run.at.isoformat())
        return f"timer set, due {due[:16]}"

    @tool
    def web_lookup(person_id: str) -> str:
        """Look this person up on the public web (LinkedIn via Bright Data) inside the Docker sandbox, in the background."""
        m = run.met.get(person_id)
        if not m:
            return "unknown person_id"
        g = m["guest"]
        run.background("lookup", pid=person_id, name=m["name"], linkedin=(g.links or {}).get("linkedin") if g else None, hint=m["org_hint"])
        return "queued; results land in memory within about a minute"

    return Agent(
        model=make_model(2500),
        callback_handler=None,
        tools=[resolve_person, recall_memory, remember_meeting, set_promise_timer, web_lookup],
        hooks=[Guardrail(run.log, met_ids=lambda: set(run.met)), Audit(run.log, "debrief agent")],
        system_prompt=SYSTEM.format(event=EVENT_NAME, now=run.at.isoformat(), style=STYLE, untrusted=UNTRUSTED),
    )


def questions(result) -> list[dict]:
    """Turn pending interrupts into questions for the page."""
    return [{"interrupt_id": i.id, "mention": (i.reason or {}).get("mention"), "candidates": (i.reason or {}).get("candidates", [])}
            for i in (result.interrupts or []) if i.name == "which_person"]
