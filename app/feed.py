"""The change feed: new internship postings from a public GitHub list, with git history as proof of when they appeared."""

import re
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/Chieler/Summer-2027-SWE-Internships"
# "New" means added in the last FEED_WINDOW_DAYS days (a weekly watch), unless the ledger stores a later ref.
DEFAULT_BASELINE = "03eb51a"  # fallback if the history has no commit old enough

RELEVANT = re.compile(r"software|engineer|developer|\bai\b|machine learning|\bml\b|data|full[- ]?stack|mobile|platform|agent", re.I)
IRRELEVANT = re.compile(r"next gen|high school|electrical|electronic|mechanical|actuator|power electronics|hardware|firmware|embedded|body controls|systems integration|reliability|manufactur|accounting|sales|marketing|finance intern|legal", re.I)
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


def baseline(repo: Path, days: int) -> str:
    """The list as it was `days` ago: the last commit before that moment."""
    ref = _git(repo, "rev-list", "-1", f"--before={days} days ago", "HEAD").strip()
    return ref[:7] if ref else DEFAULT_BASELINE


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


SOFTWARE = re.compile(r"software|\bai\b|machine learning|\bml\b|full[- ]?stack|agent|platform|product(s)? engineer", re.I)


def rank(postings: list[dict]) -> list[dict]:
    """De-duplicate by URL (the list repeats rows across sections) and put software/AI roles first."""
    seen, out = set(), []
    for p in postings:
        key = p.get("url") or (p["company"], p["role"])
        if key not in seen:
            seen.add(key)
            out.append(p)
    return sorted(out, key=lambda p: 0 if SOFTWARE.search(p["role"]) else 1)


def first_seen(repo: Path, posting: dict) -> str | None:
    """Commit (hash + date) that first added this exact posting URL: the proof of when it appeared."""
    if not posting.get("url"):
        return None
    out = _git(repo, "log", "--reverse", "--format=%h %cI", "-S", posting["url"], "--", "README.md").splitlines()
    return out[0] if out else None
