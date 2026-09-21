"""No-key check of the auto-forget mechanics, using made-up people only. Not product code.

1. Load a fake room into a temporary dataset.
2. "Meet" one person: copy them into the permanent dataset with a first-hand note.
3. forget(dataset=room) at "midnight".
4. The met person survives; the room is gone.

Run from the project root:  .venv/bin/python smoke/memory_mechanics.py
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
os.environ.pop("ANTHROPIC_BASE_URL", None)
# Local embeddings so this runs without any key (same values as .env.example).
os.environ.setdefault("EMBEDDING_PROVIDER", "fastembed")
os.environ.setdefault("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
os.environ.setdefault("EMBEDDING_DIMENSIONS", "384")

import cognee  # noqa: E402
from cognee.modules.search.types import SearchType  # noqa: E402

ROOM, PEOPLE = "room-smoke", "people-smoke"
GUESTS = [
    "Guest: Test Person Alpha. Luma bio: builds voice agents for clinics.",
    "Guest: Test Person Beta. Luma bio: angel investor in marketplaces.",
    "Guest: Test Person Gamma. Luma bio: DevRel for a database company.",
]
MET = "Met Test Person Beta at 6:10 p.m. She invests in marketplaces and asked for the StylesGo deck by Friday."


async def chunks(query, dataset):
    try:
        res = await cognee.recall(query, query_type=SearchType.CHUNKS, datasets=[dataset])
        return [str(r)[:90] for r in res]
    except Exception as e:  # noqa: BLE001 - an empty or deleted dataset may raise
        return [f"<{type(e).__name__}>"]


async def main():
    cognee.config.system_root_directory(str(ROOT / ".cognee_system"))
    cognee.config.data_root_directory(str(ROOT / ".data_storage"))

    await cognee.remember(GUESTS, dataset_name=ROOM)
    await cognee.remember([GUESTS[1], MET], dataset_name=PEOPLE)
    print("before midnight, room :", await chunks("marketplace investor", ROOM))
    print("before midnight, people:", await chunks("marketplace investor", PEOPLE))

    await cognee.forget(dataset=ROOM)
    print("after midnight, room  :", await chunks("marketplace investor", ROOM))
    print("after midnight, people:", await chunks("marketplace investor", PEOPLE))

    await cognee.forget(dataset=PEOPLE)


asyncio.run(main())
