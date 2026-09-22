"""The brain, backed by Cognee. Three datasets:

- ME_DATASET: goals, offers, stories.
- ROOM_DATASET: tonight's guest list. Temporary; forget() at midnight wipes it.
- PEOPLE_DATASET: people you actually met. Permanent; each person can be forgotten on request.
"""

from pathlib import Path

from app.config import ME_DATASET, PEOPLE_DATASET, ROOM_DATASET, init_cognee


def _text(entry) -> str:
    for attr in ("answer", "text", "content", "search_result"):
        v = getattr(entry, attr, None) or (entry.get(attr) if isinstance(entry, dict) else None)
        if v:
            return v if isinstance(v, str) else str(v)
    return str(entry)


async def remember(docs: list[str] | str, dataset: str, background: bool = False):
    cognee = await init_cognee()
    return await cognee.remember(docs, dataset_name=dataset, run_in_background=background)


async def recall(query: str, datasets: list[str], context_only: bool = False) -> str:
    """Answer (or raw context) from the named datasets; empty string if nothing is there yet."""
    cognee = await init_cognee()
    try:
        res = await cognee.recall(query, datasets=datasets, only_context=context_only, top_k=10)
    except Exception as e:  # noqa: BLE001 - a missing or empty dataset should not break a card
        return f"(nothing recalled: {type(e).__name__})"
    return "\n".join(_text(r) for r in res)


async def load_me(me_text: str):
    return await remember(me_text, ME_DATASET)


async def load_room(guest_docs: list[str], background: bool = True):
    return await remember(guest_docs, ROOM_DATASET, background=background)


async def _items(dataset: str) -> list:
    cognee = await init_cognee()
    ds_id = await _dataset_id(dataset)
    return list(await cognee.datasets.list_data(ds_id)) if ds_id else []


async def promote(docs: list[str]) -> list[str]:
    """Write a met person's records into permanent memory. Returns the Cognee data ids this write created
    (diffed before/after; callers hold the write lock), so one person can later be forgotten exactly."""
    before = {str(i.id) for i in await _items(PEOPLE_DATASET)}
    await remember(docs, PEOPLE_DATASET)
    return sorted({str(i.id) for i in await _items(PEOPLE_DATASET)} - before)


async def forget_room():
    cognee = await init_cognee()
    return await cognee.forget(dataset=ROOM_DATASET)


async def improve_people() -> bool:
    """Cognee's self-improvement pass over permanent memory: merge duplicates, reweight, derive facts."""
    cognee = await init_cognee()
    try:
        await cognee.improve(dataset=PEOPLE_DATASET)
        return True
    except Exception:  # noqa: BLE001 - an empty dataset has nothing to improve
        return False


async def _dataset_id(name: str):
    cognee = await init_cognee()
    for ds in await cognee.datasets.list_datasets():
        if getattr(ds, "name", None) == name:
            return ds.id
    return None


def _item_text(item) -> str:
    loc = str(getattr(item, "raw_data_location", "") or "")
    try:
        return Path(loc.removeprefix("file://")).read_text(errors="ignore") if loc.startswith("file://") else ""
    except OSError:
        return ""


async def forget_person(data_ids: list[str], name: str | None = None) -> int:
    """Delete exactly this person's documents from permanent memory: by recorded data id, or, for records written
    before ids were tracked, by documents that mention their name."""
    cognee = await init_cognee()
    items, wanted = await _items(PEOPLE_DATASET), set(data_ids or [])
    targets = [i for i in items if str(i.id) in wanted]
    if not targets and name:
        targets = [i for i in items if name.lower() in _item_text(i).lower()]
    for item in targets:
        await cognee.forget(data_id=item.id, dataset=PEOPLE_DATASET)
    return len(targets)


async def graph_html(path: Path, dataset: str = PEOPLE_DATASET) -> Path:
    """Cognee's interactive graph of one dataset (access control requires naming it)."""
    cognee = await init_cognee()
    await cognee.visualize_graph(destination_file_path=str(path), dataset=dataset, full=True, max_nodes=400)
    return path
