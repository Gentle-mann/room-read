"""The relationship map: your memory as a readable graph.

Built from exact records (who you met, where they work and how we know, what you promised, what changed), with an
optional layer of the facts Cognee extracted. Every edge says what it means and where it came from.
"""

import re

from app.config import EVENT_ID, EVENT_NAME, PEOPLE_DATASET, init_cognee

EX_EMPLOYER = re.compile(r"\bex-([A-Z][\w&.\-]*)", re.I)
ROLE_AT = re.compile(r"role:\s*(.+?)\s+at\s+([^\n(]+)", re.I)
MET_LINE = re.compile(r"met:\s*(.+)", re.I)
HIDDEN_TYPES = {"DocumentChunk", "TextDocument", "TextSummary", "EntityType", "Timestamp", "NodeSet"}


def _short(s: str, n: int) -> str:
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


class MapBuilder:
    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.edges: list[dict] = []

    def node(self, nid: str, label: str, group: str, detail: dict | None = None) -> str:
        if nid not in self.nodes:
            self.nodes[nid] = {"id": nid, "label": label, "group": group, "detail": detail or {}}
        elif detail:
            self.nodes[nid]["detail"].update(detail)
        return nid

    def edge(self, a: str, b: str, label: str, source: str, note: str = ""):
        key = (a, b, label)
        if not any((e["from"], e["to"], e["label"]) == key for e in self.edges):
            self.edges.append({"from": a, "to": b, "label": label, "source": source, "note": note})

    def org(self, name: str) -> str:
        name = name.strip().rstrip(".,")
        return self.node(f"org:{name.lower()}", name, "company", {"kind": "Company"})


def build(ledger, watch: dict | None) -> MapBuilder:
    m = MapBuilder()
    me = m.node("me", "You", "you", {"kind": "You", "text": "Everything on this map is connected to you."})
    promises = ledger.promises()
    for p in ledger.people():
        ints = ledger.interactions(p["id"])
        card = next((i["card"] for i in ints if i.get("card")), None)
        tonight = p["event_id"] == EVENT_ID
        prof = p.get("profile") or {}
        detail = {
            "kind": "Person you met tonight" if tonight else "Person you met before",
            "headline": (card or {}).get("headline") or prof.get("headline") or "",
            "why": [f"{r['point']} ({r['source']})" for r in (card or {}).get("why_it_matters", [])],
            "follow_up": (card or {}).get("follow_up", ""),
            "links": p.get("links") or {},
            "web": {k: prof.get(k) for k in ("current_company", "city", "education", "retrieved_at", "notes") if prof.get(k)},
        }
        pid = m.node(p["id"], p["name"], "met-tonight" if tonight else "met-before", detail)

        # How you know them
        if tonight:
            when = (p.get("first_met_at") or "")[11:16]
            m.edge(me, pid, f"met tonight {when}".strip(), "first-hand", f"At {EVENT_NAME}; from your spoken debrief.")
        else:
            notes = " ".join(i.get("raw_text") or "" for i in ints)
            met = MET_LINE.search(notes)
            label = re.sub(r"\bme\b", "you", met.group(1).split(":")[0]) if met else "met before"  # notes are first-person
            m.edge(me, pid, _short(label, 38), "my notes", f"From your notes: {met.group(1) if met else 'an earlier meeting'}")

        # Where they work, and how we know
        notes = " ".join(i.get("raw_text") or "" for i in ints)
        role = ROLE_AT.search(notes)
        if prof.get("current_company"):
            m.edge(pid, m.org(prof["current_company"]), "works at", "web", f"Public LinkedIn via Bright Data, {str(prof.get('retrieved_at', ''))[:10]}.")
        elif role:
            m.edge(pid, m.org(role.group(2)), f"{_short(role.group(1), 20)} at", "my notes", "From your notes.")
        for job in prof.get("experience") or []:
            if job.get("company"):
                m.edge(pid, m.org(job["company"]), "used to work at", "web", f"Public LinkedIn via Bright Data ({job.get('start')}–{job.get('end')}).")
        for ex in EX_EMPLOYER.findall(notes):
            oid = m.org(ex)
            if not any(e["from"] == pid and e["to"] == oid for e in m.edges):
                m.edge(pid, oid, "used to work at", "my notes", "Only in your notes; the public profile doesn't confirm it.")

        # What you owe them
        for pr in [x for x in promises if x["person_id"] == p["id"]]:
            done = pr["status"] != "open"
            prn = m.node(f"pr:{pr['id']}", _short(pr["text"], 34), "promise-done" if done else "promise",
                         {"kind": "Promise", "text": pr["text"], "due": pr["due_at"], "status": pr["status"]})
            m.edge(prn, pid, "kept" if done else f"you owe · due {pr['due_at'][5:10]}", "first-hand", "A promise you made in a debrief.")

    # What changed in the world
    for path in (watch or {}).get("paths", []):
        live = (watch or {}).get("live_check") or {}
        company = live.get("company") or next((w for w in re.findall(r"[A-Z][\w]+", path.get("why_now", "")) if w.lower() in {n["label"].lower() for n in m.nodes.values()}), None)
        role = live.get("role") or path.get("why_now", "")
        job = m.node(f"job:{_short(role, 60)}", _short(role.replace("Internship - ", ""), 40), "posting",
                     {"kind": "New posting", "text": path.get("why_now"), "ask": path.get("ask"), "draft": path.get("draft"),
                      "live": f"Confirmed live {live.get('retrieved_at', '')[:16]} via {live.get('via')}" if live else ""})
        if company:
            m.edge(m.org(company), job, "posted this week", "feed", "From the public internship list; git history shows when it appeared.")
        if path.get("person_id") in m.nodes:
            m.edge(path["person_id"], job, "warm path", "warm", path.get("ask", ""))
    return m


def _humanize(rel: str) -> str:
    return re.sub(r"[_\-]+", " ", str(rel or "related to")).strip().lower()


async def add_cognee_layer(m: MapBuilder, limit: int = 120) -> int:
    """Overlay the entities and relationships Cognee extracted from permanent memory."""
    cognee = await init_cognee()
    from cognee.api.v1.visualize.visualize import fetch_dataset_graph_data

    ds = next((d for d in await cognee.datasets.list_datasets() if d.name == PEOPLE_DATASET), None)
    if not ds:
        return 0
    data = await fetch_dataset_graph_data(ds, full=True, max_nodes=400)
    nodes, edges = (data[0], data[1]) if isinstance(data, (tuple, list)) else (data.get("nodes", []), data.get("edges", []))

    def unpack_node(n):
        if isinstance(n, (tuple, list)):
            return str(n[0]), (n[1] if len(n) > 1 and isinstance(n[1], dict) else {})
        return str(n.get("id")), n

    people = {v["label"].lower(): k for k, v in m.nodes.items() if v["group"] in ("met-tonight", "met-before")}
    orgs = {v["label"].lower(): k for k, v in m.nodes.items() if v["group"] == "company"}
    idmap, added = {}, 0
    for n in nodes:
        nid, props = unpack_node(n)
        kind = props.get("type") or props.get("node_type") or ""
        name = props.get("name") or props.get("label") or ""
        if kind in HIDDEN_TYPES or not name or added >= limit:
            continue
        key = name.lower()
        idmap[nid] = people.get(key) or orgs.get(key) or m.node(f"cg:{nid}", _short(name, 28), "cognee",
                                                                {"kind": "Fact Cognee extracted", "text": props.get("description", "")})
        added += 1
    for e in edges:
        if isinstance(e, (tuple, list)):
            a, b, rel = str(e[0]), str(e[1]), (e[2] if len(e) > 2 else "")
        else:
            a, b, rel = str(e.get("source")), str(e.get("target")), e.get("relationship_name") or e.get("label", "")
        if a in idmap and b in idmap and idmap[a] != idmap[b]:
            m.edge(idmap[a], idmap[b], _humanize(rel), "cognee", "Relationship Cognee extracted from your memory.")
    return added
