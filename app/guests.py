"""The room: the event's guest list as a name dictionary, plus name resolution."""

import json
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

FIELDS = ["id", "name", "bio", "linkedin", "x", "instagram", "tiktok", "youtube", "website", "username"]
# Placeholder handles people typed instead of leaving the field blank.
JUNK_HANDLES = {"in", "i", "a", "no_account", "no_instagram", "notwitter", "donot", "idontuseit", "www.linkedin.com", "ww.instagram.com", "no_using_instagram"}


@dataclass
class Guest:
    id: str
    name: str
    bio: str = ""
    linkedin: str = ""
    x: str = ""
    instagram: str = ""
    tiktok: str = ""
    youtube: str = ""
    website: str = ""
    username: str = ""
    role: str = "guest"  # guest | host
    aliases: list[str] = field(default_factory=list)  # other Luma ids merged into this person

    @property
    def linkedin_url(self) -> str:
        return f"https://www.linkedin.com{self.linkedin}" if self.linkedin.startswith("/") else ""

    @property
    def links(self) -> dict:
        out = {}
        if self.linkedin_url:
            out["linkedin"] = self.linkedin_url
        if self.x and self.x.lower() not in JUNK_HANDLES:
            out["x"] = f"https://x.com/{self.x}"
        if self.website:
            out["website"] = self.website
        return out

    def as_doc(self, event_name: str) -> str:
        """Text Cognee remembers for this guest. Only what they published on Luma."""
        parts = [f"{self.name} is registered for {event_name}" + (" as a host." if self.role == "host" else ".")]
        if self.bio:
            parts.append(f"Their Luma bio says: {self.bio}")
        if self.links:
            parts.append("Public links they added: " + ", ".join(self.links.values()))
        parts.append(f"(source: Luma guest list, guest id {self.id})")
        return " ".join(parts)

    def public(self) -> dict:
        d = asdict(self)
        d["links"] = self.links
        return d


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.casefold().replace(".", " ").split())


def _rows_from_dir(folder: Path) -> list[list[str]]:
    rows = []
    for f in sorted(folder.glob("luma_*.json")):
        for line in json.loads(f.read_text()).split("\n"):
            if line.strip():
                rows.append(line.split("\t"))
    return rows


def load_guests(folder: Path | None, hosts: list[str] | None = None) -> list[Guest]:
    """Load the exported guest list, merging duplicate registrations that share a LinkedIn handle."""
    guests: list[Guest] = []
    by_linkedin: dict[str, Guest] = {}
    if folder and folder.exists():
        for row in _rows_from_dir(folder):
            row = (row + [""] * len(FIELDS))[: len(FIELDS)]
            g = Guest(**dict(zip(FIELDS, row)))
            key = g.linkedin.lower().rstrip("/")
            if key and key in by_linkedin:
                twin = by_linkedin[key]
                twin.aliases.append(g.id)
                for f in ("bio", "x", "instagram", "website", "username"):
                    if not getattr(twin, f) and getattr(g, f):
                        setattr(twin, f, getattr(g, f))
                continue
            if key:
                by_linkedin[key] = g
            guests.append(g)
    for name in hosts or []:
        match = [g for g in guests if norm(g.name) == norm(name)]
        if match:
            match[0].role = "host"
        else:
            guests.append(Guest(id=f"host-{norm(name).replace(' ', '-')}", name=name, role="host"))
    return guests


@dataclass
class Resolution:
    status: str  # unique | ambiguous | none
    mention: str
    matches: list[Guest]


def resolve(mention: str, guests: list[Guest]) -> Resolution:
    """Match a spoken name to the guest list. Exact full name wins; otherwise every token must prefix a name token."""
    m = norm(mention)
    if not m:
        return Resolution("none", mention, [])
    exact = [g for g in guests if norm(g.name) == m]
    if len(exact) == 1:
        return Resolution("unique", mention, exact)
    tokens = m.split()
    hits = []
    for g in guests:
        name_tokens = norm(g.name).split()
        if all(any(nt.startswith(t) for nt in name_tokens) for t in tokens):
            # The first spoken token must match the start of some name token exactly as a word prefix.
            hits.append(g)
    # Prefer whole-word matches ("Sam" should not pull in "Samantha" when a "Sam" exists).
    whole = [g for g in hits if all(t in norm(g.name).split() for t in tokens)]
    pool = whole or hits
    if len(pool) == 1:
        return Resolution("unique", mention, pool)
    if pool:
        return Resolution("ambiguous", mention, pool[:8])
    return Resolution("none", mention, [])
