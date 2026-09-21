"""Tests for the parts that need no model: name resolution, feed parsing, ledger. Made-up people only."""

import json

from app.feed import is_relevant, parse_row
from app.guests import load_guests, resolve
from app.ledger import Ledger


def make_room(tmp_path):
    rows = [
        "usr-1\tKiran Alvarez\t\t/in/kiran-a\tkirana\t\t\t\t\t",
        "usr-2\tKiran Bose\tAngel investor\t/in/kiran-b\t\t\t\t\t\t",
        "usr-3\tKiran Cho\t\t\t\t\t\t\t\t",
        "usr-4\tPriya Natarajan\tBuilding an event app\t/in/priya-n\t\t\t\t\thttps://example.com\t",
        "usr-5\tMorgan Reyes\t\t/in/morgan-r\tin\tmorganr\t\t\t\t",
        "usr-6\tMorgan Reyes\t\t/in/morgan-r\t\t\t\t\t\t",
        "usr-7\tJosé Núñez\t\t/in/jose-n\t\t\t\t\t\t",
        "usr-8\tTheo\t\t/in/theo-a\t\t\t\t\t\t",
        "usr-9\tTheodora Pike\t\t/in/theo-p\t\t\t\t\t\t",
    ]
    (tmp_path / "luma_a.json").write_text(json.dumps("\n".join(rows)))
    return load_guests(tmp_path, hosts=["Vasilije Markovic"])


def test_unique_first_name(tmp_path):
    r = resolve("Priya", make_room(tmp_path))
    assert r.status == "unique" and r.matches[0].name == "Priya Natarajan"


def test_ambiguous_first_name_asks(tmp_path):
    r = resolve("kiran", make_room(tmp_path))
    assert r.status == "ambiguous" and {g.name for g in r.matches} == {"Kiran Alvarez", "Kiran Bose", "Kiran Cho"}


def test_first_name_plus_initial(tmp_path):
    r = resolve("Kiran C", make_room(tmp_path))
    assert r.status == "unique" and r.matches[0].name == "Kiran Cho"


def test_accents_and_whole_words(tmp_path):
    room = make_room(tmp_path)
    assert resolve("jose nunez", room).matches[0].name == "José Núñez"
    assert resolve("Theo", room).matches[0].name == "Theo"  # does not become ambiguous with Theodora


def test_duplicate_registrations_merge(tmp_path):
    room = make_room(tmp_path)
    vals = [g for g in room if g.name == "Morgan Reyes"]
    assert len(vals) == 1 and vals[0].aliases == ["usr-6"]
    assert "x" not in vals[0].links  # "in" was a placeholder, not a handle


def test_unknown_name_and_host(tmp_path):
    room = make_room(tmp_path)
    assert resolve("Zelda", room).status == "none"
    host = resolve("Vasilije", room)
    assert host.status == "unique" and host.matches[0].role == "host"


def test_feed_row_parsing_and_relevance():
    row = parse_row("| Tesla | Internship - Software Engineering - People Products - Winter/Spring 2027 | 2026-09-18 | — | [Apply](https://www.tesla.com/careers/search/job/284003) |")
    assert row == {
        "company": "Tesla",
        "role": "Internship - Software Engineering - People Products - Winter/Spring 2027",
        "posted": "2026-09-18",
        "url": "https://www.tesla.com/careers/search/job/284003",
    }
    assert is_relevant(row)
    assert not is_relevant({"role": "Next Gen Internship, Software Engineer (high school)"})
    assert not is_relevant({"role": "Internship - Electrical Engineer - Power Electronics"})
    assert parse_row("| Company | Role | Posted | Applied | Link |") is None


def test_ledger_promises_and_forget(tmp_path):
    led = Ledger(tmp_path / "l.db")
    pid = led.upsert_person(name="Test Person", guest_id="usr-1", source="debrief", met_at="2026-09-21T18:10", event_id="e")
    assert led.upsert_person(name="Test Person", guest_id="usr-1", source="debrief", met_at="x", event_id="e") == pid
    pr = led.add_promise(person_id=pid, text="send deck", due_at="2026-09-22T09:00", created_at="now")
    assert led.promises("open")[0]["person_name"] == "Test Person"
    led.close_promise(pr, "kept", "later")
    assert led.promises("open") == []
    led.delete_person(pid)
    assert led.people() == [] and led.promises() == []
