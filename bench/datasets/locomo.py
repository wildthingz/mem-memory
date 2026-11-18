"""LoCoMo (Maharana et al., ACL 2024, arXiv 2402.17753): the official `locomo10.json`, pinned.

Source: github.com/snap-research/locomo, `data/locomo10.json` at the commit below; pinned by sha256
and size like the LongMemEval_S file (`bench.datasets.fetch`). Licence: CC BY-NC 4.0 (the repo's
LICENSE.txt; non-commercial use only; see bench/datasets/README.md). The file is not checked in.

Layout: a JSON list of 10 conversations. Each has `conversation` (`speaker_a`, `speaker_b`,
`session_N` = list of turns {speaker, dia_id, text, blip_caption?}, `session_N_date_time` =
"1:56 pm on 8 May, 2023") and `qa` (question, answer, evidence, category; category 5 has
`adversarial_answer` instead of `answer`). Categories in the file: 1 multi-hop, 2 temporal,
3 open-domain, 4 single-hop, 5 adversarial (the order of evaluation.py's branches, not the paper's
table). 1,986 questions in all (1,540 in categories 1-4).

Round mapping (the memory pipeline ingests (user, assistant) rounds): `speaker_a` plays the user, a
round starts at each speaker_a turn and takes the speaker_b turns that follow; a speaker_b turn
before the first speaker_a turn is a round with an empty user side. Every turn is written
"<speaker>: <text>", an image turn adds " [image: <blip_caption>]".
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

NAME = "LoCoMo (locomo10)"
FILE = "locomo10.json"
COMMIT = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"  # snap-research/locomo main, 2026-10-26
URL = f"https://raw.githubusercontent.com/snap-research/locomo/{COMMIT}/data/{FILE}"
SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
BYTES = 2_805_274
LICENSE = "CC BY-NC 4.0"
CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
ADVERSARIAL = 5
_DATE = re.compile(r"(\d{1,2}):(\d\d) (am|pm) on (\d{1,2}) (\w+), (\d{4})")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(path: Path) -> str:
    """The file's sha256 after checking size and digest against the pin."""
    if (size := Path(path).stat().st_size) != BYTES:
        raise ValueError(f"{path}: {size} bytes, the pinned {FILE} has {BYTES} bytes")
    digest = sha256_file(path)
    if digest != SHA256:
        raise ValueError(f"{path}: sha256 {digest} != the pinned {SHA256}")
    return digest


def fetch(dest: Path) -> str:
    """Download the pinned file into `dest` and verify it (a mismatch deletes nothing and raises)."""
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / FILE
    with urllib.request.urlopen(URL, timeout=60) as r:
        path.write_bytes(r.read())
    return verify(path)


def parse_date(raw: str) -> str:
    """'1:56 pm on 8 May, 2023' -> '2023-05-08T13:56:00Z' (UTC; the file has no zone)."""
    m = _DATE.fullmatch(raw.strip())
    if not m:
        raise ValueError(f"unrecognised LoCoMo session date {raw!r}")
    hh, mm, ap, day, month, year = m.groups()
    t = datetime.strptime(f"{hh}:{mm} {ap} {day} {month} {year}", "%I:%M %p %d %B %Y")
    return t.strftime("%Y-%m-%dT%H:%M:00Z")


@dataclass(frozen=True)
class Session:
    id: str  # "D<N>", the prefix of the file's dia_ids
    date: str  # ISO UTC
    rounds: list[tuple[str, str]]


@dataclass(frozen=True)
class QA:
    index: int
    question: str
    answer: str | None  # str() of the file's value; None for the adversarial category
    category: int
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class Conversation:
    sample_id: str
    sessions: tuple[Session, ...]
    qa: tuple[QA, ...]

    @property
    def as_of(self) -> str:
        """The day after the last session: when the questions are asked (the file has no date)."""
        last = datetime.strptime(self.sessions[-1].date, "%Y-%m-%dT%H:%M:00Z")
        return (last + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:00Z")


def _turn(t: dict[str, Any]) -> str:
    text = f"{t['speaker']}: {t['text']}"
    return f"{text} [image: {t['blip_caption']}]" if t.get("blip_caption") else text


def rounds_of(turns: list[dict[str, Any]], speaker_a: str) -> list[tuple[str, str]]:
    out: list[list[str]] = []
    for t in turns:
        if t["speaker"] == speaker_a or not out:
            out.append(["", ""])
            slot = 0 if t["speaker"] == speaker_a else 1
        else:
            slot = 1
        cur = out[-1]
        cur[slot] = f"{cur[slot]}\n{_turn(t)}" if cur[slot] else _turn(t)
    return [(u, a) for u, a in out]


def conversation(entry: dict[str, Any]) -> Conversation:
    c = entry["conversation"]
    nums = sorted(int(k.split("_")[1]) for k in c if re.fullmatch(r"session_\d+", k))
    sessions = tuple(
        Session(f"D{n}", parse_date(c[f"session_{n}_date_time"]),
                rounds_of(c[f"session_{n}"], c["speaker_a"]))
        for n in nums
    )  # fmt: skip
    qa = tuple(
        QA(i, q["question"], None if q["category"] == ADVERSARIAL else str(q["answer"]),
           int(q["category"]), tuple(q.get("evidence", ())))
        for i, q in enumerate(entry["qa"])
    )  # fmt: skip
    return Conversation(entry["sample_id"], sessions, qa)


def load(path: Path) -> list[Conversation]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"{path}: expected a JSON list of conversations")
    convs = [conversation(e) for e in data]
    ids = [c.sample_id for c in convs]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate sample ids")
    return convs
