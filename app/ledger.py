"""Exact records the brain acts on: who you met, promises with due times, drafts, feed state.

Cognee holds the knowledge (what people said, who they are, how things connect). This ledger holds the
things that must be exact: timestamps, due dates, and statuses.
"""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS people (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, guest_id TEXT, links TEXT, bio TEXT,
  source TEXT, first_met_at TEXT, event_id TEXT, content_hashes TEXT DEFAULT '[]', profile TEXT
);
CREATE TABLE IF NOT EXISTS interactions (
  id TEXT PRIMARY KEY, person_id TEXT, at TEXT, event_id TEXT, raw_text TEXT, summary TEXT, card TEXT
);
CREATE TABLE IF NOT EXISTS promises (
  id TEXT PRIMARY KEY, person_id TEXT, text TEXT, due_at TEXT, status TEXT DEFAULT 'open', created_at TEXT, closed_at TEXT
);
CREATE TABLE IF NOT EXISTS drafts (
  id TEXT PRIMARY KEY, person_id TEXT, kind TEXT, text TEXT, reason TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT);
"""


class Ledger:
    def __init__(self, path: Path):
        self.path = path
        with self.db() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def db(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # --- people ---
    def upsert_person(self, *, name, guest_id=None, links=None, bio=None, source, met_at, event_id, person_id=None) -> str:
        with self.db() as c:
            row = None
            if guest_id:
                row = c.execute("SELECT id FROM people WHERE guest_id=?", (guest_id,)).fetchone()
            if not row:
                row = c.execute("SELECT id FROM people WHERE lower(name)=lower(?) AND guest_id IS NULL", (name,)).fetchone()
            if row:
                return row["id"]
            pid = person_id or f"p-{uuid.uuid4().hex[:8]}"
            c.execute(
                "INSERT INTO people (id,name,guest_id,links,bio,source,first_met_at,event_id) VALUES (?,?,?,?,?,?,?,?)",
                (pid, name, guest_id, json.dumps(links or {}), bio or "", source, met_at, event_id),
            )
            return pid

    def people(self) -> list[dict]:
        with self.db() as c:
            return [self._person(r) for r in c.execute("SELECT * FROM people ORDER BY first_met_at DESC")]

    def person(self, pid: str) -> dict | None:
        with self.db() as c:
            r = c.execute("SELECT * FROM people WHERE id=?", (pid,)).fetchone()
            return self._person(r) if r else None

    def add_content_hashes(self, pid: str, hashes: list[str]):
        p = self.person(pid)
        merged = sorted(set((p or {}).get("content_hashes", []) + [h for h in hashes if h]))
        with self.db() as c:
            c.execute("UPDATE people SET content_hashes=? WHERE id=?", (json.dumps(merged), pid))

    def set_profile(self, pid: str, profile: dict):
        with self.db() as c:
            c.execute("UPDATE people SET profile=? WHERE id=?", (json.dumps(profile), pid))

    def delete_person(self, pid: str):
        with self.db() as c:
            for table, col in (("interactions", "person_id"), ("promises", "person_id"), ("drafts", "person_id"), ("people", "id")):
                c.execute(f"DELETE FROM {table} WHERE {col}=?", (pid,))

    @staticmethod
    def _person(r) -> dict:
        d = dict(r)
        d["links"] = json.loads(d.get("links") or "{}")
        d["content_hashes"] = json.loads(d.get("content_hashes") or "[]")
        d["profile"] = json.loads(d["profile"]) if d.get("profile") else None
        return d

    # --- interactions ---
    def add_interaction(self, *, person_id, at, event_id, raw_text, summary, card=None) -> str:
        iid = f"i-{uuid.uuid4().hex[:8]}"
        with self.db() as c:
            c.execute(
                "INSERT INTO interactions VALUES (?,?,?,?,?,?,?)",
                (iid, person_id, at, event_id, raw_text, summary, json.dumps(card) if card else None),
            )
        return iid

    def set_card(self, iid: str, card: dict):
        with self.db() as c:
            c.execute("UPDATE interactions SET card=? WHERE id=?", (json.dumps(card), iid))

    def interactions(self, person_id: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM interactions", ()
        if person_id:
            q, args = q + " WHERE person_id=?", (person_id,)
        with self.db() as c:
            rows = [dict(r) for r in c.execute(q + " ORDER BY at DESC", args)]
        for r in rows:
            r["card"] = json.loads(r["card"]) if r.get("card") else None
        return rows

    # --- promises ---
    def add_promise(self, *, person_id, text, due_at, created_at) -> str:
        pid = f"pr-{uuid.uuid4().hex[:8]}"
        with self.db() as c:
            c.execute(
                "INSERT INTO promises (id,person_id,text,due_at,status,created_at) VALUES (?,?,?,?, 'open', ?)",
                (pid, person_id, text, due_at, created_at),
            )
        return pid

    def promises(self, status: str | None = None) -> list[dict]:
        q = "SELECT p.*, pe.name AS person_name FROM promises p LEFT JOIN people pe ON pe.id=p.person_id"
        args = ()
        if status:
            q, args = q + " WHERE p.status=?", (status,)
        with self.db() as c:
            return [dict(r) for r in c.execute(q + " ORDER BY p.due_at", args)]

    def close_promise(self, pid: str, status: str, at: str):
        with self.db() as c:
            c.execute("UPDATE promises SET status=?, closed_at=? WHERE id=?", (status, at, pid))

    # --- drafts ---
    def add_draft(self, *, person_id, kind, text, reason, created_at) -> str:
        did = f"d-{uuid.uuid4().hex[:8]}"
        with self.db() as c:
            c.execute("INSERT INTO drafts VALUES (?,?,?,?,?,?)", (did, person_id, kind, text, json.dumps(reason), created_at))
        return did

    def drafts(self) -> list[dict]:
        with self.db() as c:
            rows = [dict(r) for r in c.execute(
                "SELECT d.*, pe.name AS person_name FROM drafts d LEFT JOIN people pe ON pe.id=d.person_id ORDER BY d.created_at DESC"
            )]
        for r in rows:
            r["reason"] = json.loads(r["reason"]) if r.get("reason") else None
        return rows

    # --- key/value state ---
    def get(self, k: str, default=None):
        with self.db() as c:
            r = c.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
        return json.loads(r["v"]) if r else default

    def put(self, k: str, v):
        with self.db() as c:
            c.execute("INSERT INTO kv VALUES (?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, json.dumps(v)))
