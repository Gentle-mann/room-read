"""Room Read server. No chat box: everything starts from a trigger (a debrief, a clock, a change in the world).

Run:  .venv/bin/uvicorn app.server:app --host 0.0.0.0 --port 8765
"""

import asyncio
import os
import re
import uuid
from datetime import datetime, timedelta

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import brain, debrief_agent, feed, mapview, memory, scout, scout_agent
from app.config import EVENT_ID, EVENT_NAME, FEEDS, GUESTS_DIR, HOSTS, PRIVATE, ROOM_DATASET, ROOT, now
from app.guard import Audit, Guardrail
from app.guests import Guest, load_guests, resolve
from app.ledger import Ledger

app = FastAPI(title="Room Read")
LED = Ledger(PRIVATE / "ledger.db")
STATE = {"guests": load_guests(GUESTS_DIR, HOSTS), "lookups": True}
PENDING: dict[str, dict] = {}  # pipeline mode: debrief_id -> {"text", "debrief", "questions", "resolved"}
AGENTS: dict[str, dict] = {}  # agent mode: debrief_id -> {"agent", "run", "questions", "answers"}
DEBRIEF_MODE = os.getenv("DEBRIEF_MODE", "agent")  # "pipeline" = the fixed-step fallback
WRITE_LOCK = asyncio.Lock()  # serialize Cognee writes
LOG: list[dict] = []  # what the agent did on its own, newest first


def log(kind: str, text: str):
    LOG.insert(0, {"at": now().isoformat(timespec="seconds"), "kind": kind, "text": text})
    del LOG[60:]


def guest_by_id(gid: str) -> Guest | None:
    return next((g for g in STATE["guests"] if g.id == gid), None)


# ---------- background memory writes ----------
async def remember_person(pid: str, docs: list[str], label: str):
    async with WRITE_LOCK:
        try:
            hashes = await memory.promote(docs)
            LED.add_content_hashes(pid, hashes)
            log("memory", f"Saved {label} to permanent memory ({len(docs)} document(s)).")
        except Exception as e:  # noqa: BLE001
            log("error", f"Memory write failed for {label}: {type(e).__name__}: {e}")


async def web_lookup(pid: str, name: str, linkedin_url: str | None, hint: str = ""):
    """The Strands scout agent looks the person up with Bright Data's MCP tools (inside the Docker sandbox); its typed
    report is written into Cognee with source and retrieval time. Falls back to the direct sandboxed call."""
    if not scout.ready() or not STATE["lookups"]:
        return
    if not linkedin_url and len(name.split()) < 2 and not hint:
        log("scout", f"Skipped the web lookup for {name}: a first name alone can't identify anyone. Add a surname or company.")
        return
    source = "web lookup (Strands scout agent → Bright Data MCP inside the Docker sandbox)"
    try:
        report = await asyncio.to_thread(scout_agent.look_up, name, linkedin_url, hint, log)
        if not report.found:
            log("scout", f"Scout agent could not confirm a profile for {name}: {report.notes[:100]}")
            return
        prof = {"name": report.name, "headline": report.headline, "current_company": report.current_company,
                "experience": [j.model_dump() for j in report.past_companies], "education": report.education,
                "city": report.city, "url": report.linkedin_url or linkedin_url, "notes": report.notes,
                "source": source, "retrieved_at": scout_agent.retrieved_at()}
    except Exception as e:  # noqa: BLE001 - never lose the lookup because the agent path failed
        log("scout", f"Scout agent failed ({type(e).__name__}); using the direct sandboxed call.")
        if not linkedin_url:
            return
        res = await asyncio.to_thread(scout.linkedin_profile, linkedin_url)
        if not res.get("ok"):
            log("scout", f"Web lookup for {name} failed: {res.get('error', '')[:120]}")
            return
        prof = dict(res["fields"], source="web lookup (Bright Data MCP, sandboxed)", retrieved_at=res["retrieved_at"])
    LED.set_profile(pid, prof)
    jobs = "; ".join(f"{e.get('title')} at {e.get('company')} ({e.get('start')}–{e.get('end') or 'present'})" for e in prof.get("experience") or [])
    doc = (f"According to {name}'s public LinkedIn profile, retrieved {prof['retrieved_at']} via Bright Data: "
           f"headline '{prof.get('headline')}', currently at {prof.get('current_company')}. "
           f"Work history: {jobs or 'not listed on the profile'}. Education: {'; '.join(map(str, prof.get('education') or [])) or 'n/a'}.")
    log("scout", f"Web profile for {name} verified ({prof.get('current_company') or 'no current company'}); writing it into Cognee.")
    await remember_person(pid, [doc], f"{name}'s web profile")


MAIN_LOOP: asyncio.AbstractEventLoop | None = None


def background(kind: str, **kw):
    """Schedule slow work the agent's tools asked for, so the card comes back fast.
    Strands runs plain tools on worker threads, so hand the coroutine to the server's loop thread-safely."""
    if kind == "remember":
        coro = remember_person(kw["pid"], kw["docs"], kw["label"])
    elif kind == "lookup":
        coro = web_lookup(kw["pid"], kw["name"], kw.get("linkedin"), kw.get("hint", ""))
    else:
        return
    asyncio.run_coroutine_threadsafe(coro, MAIN_LOOP)


# ---------- debrief ----------
class DebriefIn(BaseModel):
    text: str


class ResolveIn(BaseModel):
    debrief_id: str
    mention: str
    guest_id: str | None = None  # None = "not on the guest list"
    interrupt_id: str | None = None  # agent mode: which paused question this answers


@app.post("/api/debrief")
async def post_debrief(body: DebriefIn):
    if not body.text.strip():
        raise HTTPException(400, "empty debrief")
    if DEBRIEF_MODE == "agent":
        did = f"db-{uuid.uuid4().hex[:6]}"
        run = debrief_agent.DebriefRun(text=body.text, ledger=LED, guests=lambda: STATE["guests"], log=log, background=background)
        agent = debrief_agent.build_agent(run)
        AGENTS[did] = {"agent": agent, "run": run, "questions": [], "answers": {}}
        log("debrief", "Debrief agent started.")
        res = await agent.invoke_async(body.text, structured_output_model=debrief_agent.DebriefOutcome)
        return await agent_step(did, res)
    d = await brain.extract_debrief(body.text)
    did = f"db-{uuid.uuid4().hex[:6]}"
    questions, resolved = [], {}
    for m in d.people:
        r = resolve(m.name, STATE["guests"])
        if r.status == "unique":
            resolved[m.name] = r.matches[0].id
        elif r.status == "ambiguous":
            questions.append({"mention": m.name, "candidates": [g.public() for g in r.matches]})
        else:
            resolved[m.name] = None
    PENDING[did] = {"text": body.text, "debrief": d, "questions": questions, "resolved": resolved}
    if questions:
        log("resolve", f"Asked which person you meant: {', '.join(q['mention'] for q in questions)}.")
        return {"status": "question", "debrief_id": did, "questions": questions}
    return await finalize(did)


@app.post("/api/resolve")
async def post_resolve(body: ResolveIn):
    if body.debrief_id in AGENTS:
        a = AGENTS[body.debrief_id]
        iid = body.interrupt_id or next((q["interrupt_id"] for q in a["questions"] if q["mention"] == body.mention), None)
        a["answers"][iid] = body.guest_id or "none"
        remaining = [q for q in a["questions"] if q["interrupt_id"] not in a["answers"]]
        if remaining:
            return {"status": "question", "debrief_id": body.debrief_id, "questions": remaining}
        responses = [{"interruptResponse": {"interruptId": k, "response": v}} for k, v in a["answers"].items()]
        a["questions"], a["answers"] = [], {}
        log("resolve", "You answered; the debrief agent resumed.")
        res = await a["agent"].invoke_async(responses, structured_output_model=debrief_agent.DebriefOutcome)
        return await agent_step(body.debrief_id, res)
    p = PENDING.get(body.debrief_id) or {}
    if not p:
        raise HTTPException(404, "unknown debrief")
    p["resolved"][body.mention] = body.guest_id
    p["questions"] = [q for q in p["questions"] if q["mention"] != body.mention]
    if p["questions"]:
        return {"status": "question", "debrief_id": body.debrief_id, "questions": p["questions"]}
    return await finalize(body.debrief_id)


async def agent_step(did: str, res):
    """Either surface the agent's paused questions, or store its cards."""
    a = AGENTS[did]
    if res.stop_reason == "interrupt":
        a["questions"] = debrief_agent.questions(res)
        log("resolve", f"Debrief agent paused to ask: which {', '.join(q['mention'] for q in a['questions'])}?")
        return {"status": "question", "debrief_id": did, "questions": a["questions"]}
    AGENTS.pop(did)
    run, outcome = a["run"], res.structured_output
    cards = []
    for po in outcome.people:
        if po.person_id not in run.met:
            continue
        iid = run.interactions.get(po.person_id) or LED.add_interaction(
            person_id=po.person_id, at=run.at.isoformat(), event_id=EVENT_ID, raw_text=run.text, summary=outcome.summary)
        LED.set_card(iid, po.card.model_dump())
        cards.append({"person": LED.person(po.person_id), "card": po.card.model_dump(), "interaction_id": iid})
    log("debrief", f"Debrief agent finished: {', '.join(c['person']['name'] for c in cards) or 'nobody identified'}.")
    return {"status": "done", "cards": cards}


async def finalize(did: str):
    p = PENDING.pop(did)
    d, at = p["debrief"], now()
    cards = []
    for m in d.people:
        g = guest_by_id(p["resolved"].get(m.name) or "")
        name = g.name if g else m.name
        pid = LED.upsert_person(
            name=name, guest_id=g.id if g else None, links=g.links if g else {}, bio=g.bio if g else "",
            source="debrief" if g else "debrief (not on the guest list)", met_at=at.isoformat(), event_id=EVENT_ID,
        )
        person = LED.person(pid)
        card = await brain.make_card(person, d.summary, d.facts)
        iid = LED.add_interaction(person_id=pid, at=at.isoformat(), event_id=EVENT_ID, raw_text=p["text"], summary=d.summary, card=card.model_dump())
        for pr in d.promises:
            due = pr.due_iso or (at + timedelta(days=1)).replace(hour=9, minute=0, second=0).isoformat()
            LED.add_promise(person_id=pid, text=pr.text, due_at=due, created_at=at.isoformat())
            log("promise", f"Promise timer set: '{pr.text}' for {name}, due {due[:16]}.")
        docs = []
        if g:
            docs.append(g.as_doc(EVENT_NAME))
        docs.append(
            f"On {at:%Y-%m-%d at %H:%M} at {EVENT_NAME}, I met {name}{f' ({m.role}, {m.org})' if m.org or m.role else ''}. "
            f"{d.summary} What I heard: {'; '.join(d.facts)}. They are looking for: {'; '.join(d.their_needs) or 'n/a'}. "
            f"I promised: {'; '.join(x.text for x in d.promises) or 'nothing'}."
        )
        asyncio.create_task(remember_person(pid, docs, name))
        asyncio.create_task(web_lookup(pid, name, (g.links or {}).get("linkedin") if g else None, m.org or ""))
        cards.append({"person": LED.person(pid), "card": card.model_dump(), "interaction_id": iid})
    log("debrief", f"Debrief processed: {', '.join(c['person']['name'] for c in cards)}.")
    return {"status": "done", "cards": cards}


# ---------- voice: the mic next to the debrief box ----------
def name_hints() -> str:
    """Names worth spelling right: people met, tonight's picks, hosts. Biases transcription toward the room."""
    names = [p["name"] for p in LED.people()] + HOSTS
    names += [p["name"] for p in (LED.get("breakdown") or {}).get("picks", [])]
    seen = list(dict.fromkeys(n for n in names if n))[:40]
    terms = "Cognee, Bright Data, Strands, AWS, Docker, StylesGo, Minerva, Room Read"
    return f"Networking notes at {EVENT_NAME}. Terms: {terms}. Names that may come up: " + ", ".join(seen) + "."


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Speech to text for a spoken debrief, using OpenAI's transcription model."""
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(501, "no OpenAI key for transcription; use your keyboard's mic")
    from openai import AsyncOpenAI

    data = await audio.read()
    if len(data) < 800:
        raise HTTPException(400, "recording too short")
    ext = {"audio/webm": "webm", "audio/mp4": "mp4", "audio/ogg": "ogg", "audio/wav": "wav", "audio/mpeg": "mp3", "audio/x-m4a": "m4a"}
    kind = (audio.content_type or "audio/webm").split(";")[0]
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    res = await client.audio.transcriptions.create(
        model=os.getenv("TRANSCRIBE_MODEL", "gpt-4o-transcribe"),
        file=(f"debrief.{ext.get(kind, 'webm')}", data, kind),
        prompt=name_hints(),
    )
    log("voice", f"Transcribed a {len(data) // 1024} KB voice note.")
    return {"text": res.text.strip()}


# ---------- setup: load me, backfill, room ----------
def parse_backfill(text: str) -> list[dict]:
    blocks = []
    for chunk in re.split(r"^## ", text, flags=re.M)[1:]:
        name, _, rest = chunk.partition("\n")
        date = re.search(r"\d{4}-\d{2}-\d{2}", rest)
        blocks.append({"name": name.strip(), "text": rest.strip(), "date": date.group(0) if date else None})
    return blocks


@app.post("/api/setup")
async def setup(room: bool = True):
    """One-time: load Me, backfill, and (optionally) tonight's room into Cognee."""
    done = []
    if not LED.get("me_loaded"):
        async with WRITE_LOCK:
            await memory.load_me((PRIVATE / "me.md").read_text())
        LED.put("me_loaded", now().isoformat())
        done.append("me")
    if not LED.get("backfill_loaded") and (PRIVATE / "backfill.md").exists():
        for b in parse_backfill((PRIVATE / "backfill.md").read_text()):
            pid = LED.upsert_person(name=b["name"], source="my notes (backfill)", met_at=b["date"] or now().isoformat(), event_id="before-tonight")
            LED.add_interaction(person_id=pid, at=b["date"] or now().isoformat(), event_id="before-tonight", raw_text=b["text"], summary=b["text"][:200])
            await remember_person(pid, [f"My notes about {b['name']} (from before tonight):\n{b['text']}"], b["name"])
            role = re.search(r"role:\s*(.+)", b["text"])
            asyncio.create_task(web_lookup(pid, b["name"], None, role.group(1) if role else ""))
        LED.put("backfill_loaded", now().isoformat())
        done.append("backfill")
    if room and not LED.get("room_loaded") and STATE["guests"]:
        async with WRITE_LOCK:
            await memory.load_room([g.as_doc(EVENT_NAME) for g in STATE["guests"]], background=True)
        LED.put("room_loaded", now().isoformat())
        log("room", f"Loaded {len(STATE['guests'])} guests into the temporary room memory ({ROOM_DATASET}).")
        done.append("room")
    return {"loaded": done}


# ---------- pre-event breakdown ----------
@app.post("/api/breakdown")
async def post_breakdown():
    b = await brain.breakdown([g.public() for g in STATE["guests"]], HOSTS)
    LED.put("breakdown", {"at": now().isoformat(), "picks": [x.model_dump() for x in b.picks]})
    log("breakdown", f"Picked {len(b.picks)} people to find tonight out of {len(STATE['guests'])}.")
    return LED.get("breakdown")


# ---------- the watch: something changed ----------
@app.post("/api/watch")
async def post_watch():
    repo = await asyncio.to_thread(feed.ensure_repo, FEEDS)
    since = LED.get("feed_ref") or await asyncio.to_thread(feed.baseline, repo, int(os.getenv("FEED_WINDOW_DAYS", "7")))
    res = await asyncio.to_thread(feed.new_postings, repo, since)
    relevant = [p for p in res["postings"] if feed.is_relevant(p)]
    people = LED.people()
    haystack = {p["id"]: (" ".join(i["raw_text"] or "" for i in LED.interactions(p["id"])) + " " + str(p.get("profile") or "")).lower() for p in people}
    def mentions(company: str, text: str) -> bool:  # whole words: "Cisco" must not match "San Francisco"
        return re.search(rf"\b{re.escape(company.lower())}\b", text) is not None

    candidates = feed.rank([p for p in relevant if any(mentions(p["company"], h) for h in haystack.values())])
    summary = {"since": since, "head": res["head"], "head_time": res["head_time"], "new_postings": len(res["postings"]),
               "relevant": len(relevant), "people_checked": len(people), "candidates": len(candidates), "paths": []}
    if candidates:
        for c in candidates[:3]:  # git log -S walks history; keep it to the top few
            c["first_seen"] = await asyncio.to_thread(feed.first_seen, repo, c)
        top = candidates[0]
        if top.get("url") and scout.ready():
            asyncio.create_task(live_check(top))  # ~60 s through the unlocker; never block the result on it
        wp = await brain.find_warm_paths(LED, candidates[:10], hooks=[Guardrail(log), Audit(log, "warm-path agent")])
        for path in wp.paths:
            LED.add_draft(person_id=path.person_id, kind="warm path", text=path.draft,
                          reason={**path.model_dump(exclude={"draft"}), "posting": next((c for c in candidates if c["company"].lower() in path.why_now.lower()), candidates[0])},
                          created_at=now().isoformat())
        summary["paths"] = [x.model_dump() for x in wp.paths]
        summary["notes"] = wp.notes
    LED.put("watch", {"at": now().isoformat(), **summary})
    log("watch", f"Checked {summary['new_postings']} new postings against {len(people)} people: {len(summary['paths'])} worth your time.")
    return LED.get("watch")


async def live_check(posting: dict):
    """Confirm the posting is live on the employer's site via the sandboxed scout, then attach the proof."""
    res = await asyncio.to_thread(scout.page, posting["url"])
    w = LED.get("watch") or {}
    if res.get("ok"):
        w["live_check"] = {"company": posting["company"], "role": posting["role"], "title": res["fields"].get("title"),
                           "retrieved_at": res["retrieved_at"], "via": "Bright Data, inside the Docker sandbox",
                           "flagged_lines": res["fields"].get("flagged_lines", 0)}
        LED.put("watch", w)
        log("scout", f"Confirmed the {posting['company']} posting is live on their site (via Bright Data, sandboxed).")
    else:
        log("scout", f"Live check of the {posting['company']} posting failed: {res.get('error', '')[:100]}")


# ---------- morning, promises, forgetting ----------
@app.post("/api/morning")
async def post_morning():
    people = [dict(p, cards=[i["card"] for i in LED.interactions(p["id"]) if i["card"]]) for p in LED.people() if p["event_id"] == EVENT_ID]
    m = await brain.morning_followups(people, LED.promises("open"))
    for f in m.followups:
        LED.add_draft(person_id=f.person_id, kind="let go" if f.let_go else "morning follow-up", text=f.draft,
                      reason={"priority": f.priority, "reason": f.reason}, created_at=now().isoformat())
    log("morning", f"Drafted {len(m.followups)} follow-ups (labelled: triggered manually, normally 9 a.m.).")
    return {"followups": [f.model_dump() for f in m.followups]}


@app.post("/api/promise/{pid}/{status}")
async def close_promise(pid: str, status: str):
    if status not in ("kept", "dropped"):
        raise HTTPException(400, "status must be kept or dropped")
    LED.close_promise(pid, status, now().isoformat())
    return {"ok": True}


async def midnight():
    async with WRITE_LOCK:
        await memory.forget_room()
        improved = await memory.improve_people()
    met = len([p for p in LED.people() if p["event_id"] == EVENT_ID])
    log("memory", "Cognee improve() ran on permanent memory: " + ("consolidated tonight's people." if improved else "nothing to consolidate."))
    forgotten = len(STATE["guests"]) - met
    STATE["guests"] = []
    LED.put("room_forgotten_at", now().isoformat())
    log("forget", f"Midnight: forgot the room ({forgotten} people you didn't meet). Kept the {met} you met.")
    return {"forgotten": forgotten, "kept": met}


@app.post("/api/lookups/{mode}")
async def set_lookups(mode: str):
    """Turn web lookups on or off, e.g. for tests or if Bright Data is slow on stage."""
    STATE["lookups"] = mode == "on"
    log("scout", f"Web lookups turned {'on' if STATE['lookups'] else 'off'}.")
    return {"lookups": STATE["lookups"]}


@app.post("/api/midnight")
async def post_midnight():
    return await midnight()


@app.post("/api/forget/{pid}")
async def forget_person(pid: str):
    p = LED.person(pid)
    if not p:
        raise HTTPException(404, "unknown person")
    async with WRITE_LOCK:
        removed = await memory.forget_person(p["content_hashes"], p["name"])  # column holds Cognee data ids
    LED.delete_person(pid)
    log("forget", f"Forgot {p['name']} on request ({removed} memory document(s) deleted).")
    return {"forgotten": p["name"], "documents": removed}


# ---------- read side ----------
@app.get("/api/state")
async def state():
    t = now()
    promises = [dict(p, overdue=p["status"] == "open" and p["due_at"] < t.isoformat()) for p in LED.promises()]
    people = []
    for p in LED.people():
        ints = LED.interactions(p["id"])
        people.append(dict(p, interactions=ints, card=next((i["card"] for i in ints if i["card"]), None),
                           promises=[x for x in promises if x["person_id"] == p["id"]]))
    return {
        "event": {"id": EVENT_ID, "name": EVENT_NAME, "room_size": len(STATE["guests"]), "room_forgotten_at": LED.get("room_forgotten_at")},
        "met_tonight": len([p for p in people if p["event_id"] == EVENT_ID]),
        "people": people, "promises": promises, "drafts": LED.drafts(),
        "breakdown": LED.get("breakdown"), "watch": LED.get("watch"), "log": LOG,
        "scout_ready": scout.ready() and STATE["lookups"], "setup": {k: LED.get(k) for k in ("me_loaded", "backfill_loaded", "room_loaded")},
    }


@app.get("/api/graph")
async def graph(dataset: str = "people"):
    """Graph view: ?dataset=people (default), me, or room."""
    name = {"people": memory.PEOPLE_DATASET, "me": memory.ME_DATASET, "room": ROOM_DATASET}.get(dataset, memory.PEOPLE_DATASET)
    path = await memory.graph_html(PRIVATE / f"graph-{dataset}.html", name)
    return FileResponse(path)


@app.get("/api/map")
async def relationship_map(cognee: int = 0):
    """Nodes and labeled edges for the map page; ?cognee=1 overlays what Cognee extracted."""
    m = mapview.build(LED, LED.get("watch"))
    facts, error = 0, None
    if cognee:
        try:
            facts = await mapview.add_cognee_layer(m)
        except Exception as e:  # noqa: BLE001 - the map still works without the overlay
            error = f"Cognee layer unavailable ({type(e).__name__})"
    return {"nodes": list(m.nodes.values()), "edges": m.edges, "cognee_facts": facts, "error": error}


@app.get("/map")
async def map_page():
    return FileResponse(ROOT / "static" / "map.html")


@app.get("/")
async def index():
    return FileResponse(ROOT / "static" / "index.html")


# ---------- the clock ----------
async def clock():
    """Real midnight auto-forget for tonight's room, plus overdue-promise nudges."""
    nudged = set()
    while True:
        t = now()
        if STATE["guests"] and not LED.get("room_forgotten_at") and t.date() > datetime.fromisoformat(f"{EVENT_ID}T00:00").date():
            await midnight()
        for p in LED.promises("open"):
            if p["due_at"] < t.isoformat() and p["id"] not in nudged:
                nudged.add(p["id"])
                log("promise", f"Overdue: '{p['text']}' for {p['person_name']}.")
        await asyncio.sleep(30)


@app.on_event("startup")
async def start_clock():
    global MAIN_LOOP
    MAIN_LOOP = asyncio.get_running_loop()
    asyncio.create_task(clock())
    log("start", f"Room Read up. {len(STATE['guests'])} people in tonight's room; {len(LED.people())} people in memory.")

