# Room Read

A personal brain that remembers everyone you meet, why they matter to your goals, and what you promised them. It has no chat box. It acts on triggers: a debrief, a change in the world, and the clock.

Built for Battle of the Personal Brains (2026-09-21) with Cognee, Bright Data, AWS Strands Agents, and Docker Sandboxes.

**Disclosure:** the setup, the core modules, and the UI were written before the 4 p.m. kickoff. The git history shows what was built when.

## How it works

| Piece | Tool | Where |
|---|---|---|
| Memory: Me, tonight's room (temporary), people you met (permanent) | Cognee, local (LadybugDB, LanceDB, SQLite) | `app/memory.py` |
| Reasoning: debrief extraction, person cards, "who to meet," warm paths, follow-ups | Strands Agents (tools plus typed output) | `app/brain.py` |
| Live web: LinkedIn profiles, job pages, search | Bright Data MCP, called **inside a Docker sandbox** | `sandbox/scout.mjs`, `app/scout.py` |
| Isolation: the only code that reads raw web text; its network reaches only `api.brightdata.com`; it can't see the brain; it returns whitelisted fields and strips injection lines | Docker Sandboxes (`sbx`, deny-all policy) | `rr-scout` sandbox |
| Exact records: promises with due times, drafts, feed state | SQLite | `app/ledger.py` |
| Change feed: new internship postings, with git history as proof of when each appeared | Public GitHub list | `app/feed.py` |

**Privacy rules:**

- The guest list is loaded into a temporary dataset, `room-<event>`.
- Only people you debrief are promoted to `people`.
- At midnight, `forget(dataset=room)` wipes everyone you didn't meet.
- "Forget" on any person deletes their documents.
- Drafts are never sent.

## Run

```bash
cd ~/Projects/room-read
cp .env.example .env    # then fill in keys; .env is gitignored
.venv/bin/uvicorn app.server:app --host 0.0.0.0 --port 8765
```

Open http://localhost:8765. On a phone on the same Wi-Fi, use http://<laptop-ip>:8765, or run `cloudflared tunnel --url http://localhost:8765` for an HTTPS link.

**Setup checks:**

```bash
.venv/bin/python -m pytest -q tests          # no keys needed
.venv/bin/python smoke/check_setup.py        # keys, agent, Cognee, Bright Data, sandbox
.venv/bin/python smoke/memory_mechanics.py   # auto-forget with made-up people
sbx exec -w /home/agent/scout rr-scout -- node scout.mjs selftest   # injection-stripping proof
```

If `sandbox/scout.mjs` changes, copy it into the sandbox again: `sbx cp sandbox/scout.mjs rr-scout:/home/agent/scout/scout.mjs`.

## Demo order

1. **Load brain.** This loads Me, the backfill (`data/private/backfill.md`), and tonight's room into Cognee. The room loads in the background.
2. **Pick who to meet.** Your 10–15 people out of the room, grouped by kind of overlap. Each pick cites its source.
3. **Just met someone?** Dictate a debrief.
   - An ambiguous name gets one question ("Which Kiran?").
   - The card shows reasons with sources, a follow-up draft, and a promise timer.
   - A sandboxed web lookup runs in the background.
4. **Check for changes.** New postings are compared against the people in memory, and the app reports how many are worth your time. The warm path: a company posts a role, nobody you know works there now, but someone you met months ago used to, and said "let's keep in touch." They get an honest ask for an intro, as a draft only.
5. **Fast-forward to midnight.** The room is forgotten, and the people you met stay.

## Files kept out of git

`.env`, `data/private/` (the Me profile, backfill, and ledger), `.cognee_system/`, `.data_storage/`, and `data/feeds/`. The guest-list export stays in the Claude session's scratch folder and is never copied here.
