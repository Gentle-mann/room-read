"""The change feed: new internship postings from a public GitHub list, with git history as proof of when they appeared."""

import re
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/Chieler/Summer-2027-SWE-Internships"
# Default baseline: the list as of 2026-09-18. "New" means added after it unless the ledger stores a later ref.
DEFAULT_BASELINE = "03eb51a"

RELEVANT = re.compile(r"software|engineer|developer|\bai\b|machine learning|\bml\b|data|full[- ]?stack|mobile|platform|agent", re.I)
IRRELEVANT = re.compile(r"next gen|high school|electrical|mechanical|power electronics|hardware|firmware|embedded|manufactur|accounting|sales|marketing|finance intern|legal", re.I)
LINK = re.compile(r"\((https?://[^)]+)\)")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True, timeout=120).stdout


def ensure_repo(folder: Path) -> Path:
    repo = folder / "chieler"
    if not (repo / ".git").exists():
        folder.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", "--filter=blob:none", REPO_URL, str(repo)], check=True, timeout=300)
    else:
        _git(repo, "pull", "--quiet", "--ff-only")
    return repo


def parse_row(line: str) -> dict | None:
    cells = [c.strip() for c in line.strip().split("|")]
    if len(cells) < 7 or cells[1] in ("Company", "---") or set(cells[1]) <= {"-"}:
        return None
    link = LINK.search(cells[-2])
    return {
        "company": cells[1],
        "role": " | ".join(cells[2:-4]),
        "posted": cells[-4],
        "url": link.group(1) if link else "",
    }


def new_postings(repo: Path, since_ref: str) -> dict:
    """Rows added to the README since since_ref, with the HEAD commit that proves it."""
    head = _git(repo, "log", "-1", "--format=%h %cI").split()
    diff = _git(repo, "diff", f"{since_ref}..HEAD", "--", "README.md")
    added = [parse_row(l[1:]) for l in diff.splitlines() if l.startswith("+|")]
    added = [a for a in added if a]
    return {"since": since_ref, "head": head[0], "head_time": head[1], "postings": added}


def is_relevant(posting: dict) -> bool:
    role = posting["role"]
    return bool(RELEVANT.search(role)) and not IRRELEVANT.search(role)


def first_seen(repo: Path, posting: dict) -> str | None:
    """Commit (hash + date) that first added this exact posting URL: the proof of when it appeared."""
    if not posting.get("url"):
        return None
    out = _git(repo, "log", "--reverse", "--format=%h %cI", "-S", posting["url"], "--", "README.md").splitlines()
    return out[0] if out else None
