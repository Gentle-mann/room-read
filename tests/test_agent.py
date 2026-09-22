"""Debrief agent wiring and Strands hooks, without calling a model. Made-up people only."""

import json
from types import SimpleNamespace

from app import debrief_agent
from app.guard import Audit, Guardrail
from app.guests import Guest
from app.ledger import Ledger


def _run(tmp_path, guests):
    logs, jobs = [], []
    run = debrief_agent.DebriefRun(
        text="Met Priya, she needs a mobile dev. I promised the API doc tomorrow.",
        ledger=Ledger(tmp_path / "l.db"),
        guests=lambda: guests,
        log=lambda k, t: logs.append((k, t)),
        background=lambda kind, **kw: jobs.append((kind, kw)),
    )
    return run, logs, jobs


def test_guardrail_blocks_outbound_and_unmet_lookups():
    logs = []
    g = Guardrail(lambda k, t: logs.append(t), met_ids=lambda: {"p-met"})
    send = SimpleNamespace(tool_use={"name": "send_email", "input": {}}, cancel_tool=False)
    g.before(send)
    assert send.cancel_tool and "never sends" in send.cancel_tool
    unmet = SimpleNamespace(tool_use={"name": "web_lookup", "input": {"person_id": "p-stranger"}}, cancel_tool=False)
    g.before(unmet)
    assert unmet.cancel_tool
    met = SimpleNamespace(tool_use={"name": "web_lookup", "input": {"person_id": "p-met"}}, cancel_tool=False)
    g.before(met)
    assert met.cancel_tool is False
    assert len(logs) == 2


def test_audit_logs_tool_calls():
    logs = []
    a = Audit(lambda k, t: logs.append((k, t)), "debrief agent")
    a.after(SimpleNamespace(tool_use={"name": "set_promise_timer", "input": {"promise": "send deck"}}, cancel_message=None,
                            result={"status": "success", "content": [{"text": "timer set"}]}))
    assert logs[0][0] == "agent" and "set_promise_timer" in logs[0][1] and "✓" in logs[0][1]


def test_agent_builds_with_tools_and_hooks(tmp_path):
    run, _, _ = _run(tmp_path, [])
    agent = debrief_agent.build_agent(run)
    assert set(agent.tool_names) == {"resolve_person", "recall_memory", "remember_meeting", "set_promise_timer", "web_lookup"}


def test_tools_resolve_remember_and_time(tmp_path):
    room = [Guest(id="usr-1", name="Priya Natarajan", bio="Builds voice agents", linkedin="/in/priya-n")]
    run, _, jobs = _run(tmp_path, room)
    agent = debrief_agent.build_agent(run)
    out = json.loads(agent.tool.resolve_person(name="Priya", record_direct_tool_call=False)["content"][0]["text"])
    assert out["on_guest_list"] and out["name"] == "Priya Natarajan"
    pid = out["person_id"]
    agent.tool.remember_meeting(person_id=pid, what_i_heard="needs a mobile dev", their_needs=["mobile dev"], i_promised=["API doc"], record_direct_tool_call=False)
    agent.tool.set_promise_timer(person_id=pid, promise="send the API doc", record_direct_tool_call=False)
    assert jobs[0][0] == "remember" and "Priya Natarajan" in jobs[0][1]["docs"][-1]
    assert run.ledger.promises("open")[0]["text"] == "send the API doc"
    assert run.interactions[pid]
