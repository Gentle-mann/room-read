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


async def promote(docs: list[str]) -> list[str]:
    """Write a met person's records into permanent memory. Returns content hashes for later forget."""
    result = await remember(docs, PEOPLE_DATASET)
    return [i.get("content_hash") for i in (getattr(result, "items", None) or []) if i.get("content_hash")]


async def forget_room():
    cognee = await init_cognee()
    return await cognee.forget(dataset=ROOM_DATASET)


async def _dataset_id(name: str):
    cognee = await init_cognee()
    for ds in await cognee.datasets.list_datasets():
        if getattr(ds, "name", None) == name:
            return ds.id
    return None


async def forget_person(content_hashes: list[str]) -> int:
    """Delete exactly this person's documents from permanent memory."""
    cognee = await init_cognee()
    ds_id = await _dataset_id(PEOPLE_DATASET)
    if not ds_id or not content_hashes:
        return 0
    wanted, removed = set(content_hashes), 0
    for item in await cognee.datasets.list_data(ds_id):
        if getattr(item, "content_hash", None) in wanted:
            await cognee.forget(data_id=item.id, dataset=PEOPLE_DATASET)
            removed += 1
    return removed


async def graph_html(path: Path) -> Path:
    cognee = await init_cognee()
    await cognee.visualize_graph(destination_file_path=str(path))
    return path
