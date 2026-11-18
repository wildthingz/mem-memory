"""Regenerate the tiny LongMemEval-layout fixtures (no download): `uv run python -m bench.fixtures.lme.make_fixture`.

`lme_s_mini.json`: 20 questions in the official `longmemeval_s_cleaned.json` layout (3 per type + 2
`_abs`), 6 sessions each (1-2 answer sessions with a `has_answer` user turn, one duplicated session
id to exercise dedup, a leading assistant turn in one session). `lme_m_mini.json`: 4 haystacks of
shared filler sessions for the L-corpus builder. Vocabulary is disjoint between question topics and
filler so a lexical fake backend can retrieve them; they stand in for the real files only in tests.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

HERE = Path(__file__).parent
TYPES = ["knowledge-update", "temporal-reasoning", "multi-session", "single-session-user",
         "single-session-assistant", "single-session-preference"]  # fmt: skip
TOPIC = ["amber", "basalt", "cobalt", "dune", "ember", "fjord", "garnet", "harbor", "indigo",
         "juniper", "kelp", "lagoon", "marble", "nectar", "onyx", "prairie", "quartz", "reef",
         "saffron", "tundra", "umber", "velvet"]  # fmt: skip
FILL = ["pasta", "bicycle", "invoice", "garden", "printer", "holiday", "recipe", "ticket", "movie",
        "budget", "laptop", "yoga", "coffee", "train", "podcast", "paint", "camera", "soccer"]  # fmt: skip


def _filler_session(rng: random.Random, k: int) -> list[dict]:
    turns = []
    for _ in range(k):
        w = rng.sample(FILL, 4)
        turns.append({"role": "user", "content": f"Can we talk about {w[0]} and {w[1]} today?"})
        turns.append(
            {"role": "assistant", "content": f"Sure, here is a note on {w[2]} and {w[3]}."}
        )
    return turns


def _answer_session(topic: str, answer: str) -> list[dict]:
    return [
        {"role": "user", "content": f"My {topic} project update: the {topic} value is {answer}.",
         "has_answer": True},
        {"role": "assistant", "content": f"Noted, your {topic} value is {answer}."},
        {"role": "user", "content": f"Also remind me about {topic} next week."},
        {"role": "assistant", "content": "I will."},
    ]  # fmt: skip


def make_s() -> list[dict]:
    rng = random.Random(7)
    out = []
    plan = [(t, False) for t in TYPES for _ in range(3)] + [(TYPES[0], True), (TYPES[3], True)]
    for n, (qtype, abs_) in enumerate(plan):
        topic = TOPIC[n]
        qid = f"{n:04x}ab{n}" + ("_abs" if abs_ else "")
        answer = f"{topic}-{100 + n}"
        ids = [f"s{n}_{j}" for j in range(6)]
        sessions = [_filler_session(rng, 3) for _ in ids]
        gold = [] if abs_ else [ids[2]]
        if not abs_:
            sessions[2] = _answer_session(topic, answer)
        if qtype == "multi-session" and not abs_:
            gold.append(ids[4])
            sessions[4] = _answer_session(topic, answer + "b")
        if n == 1:
            sessions[0] = [{"role": "assistant", "content": "Welcome back."}, *sessions[0]]
        ids_h = list(ids)
        sess_h = list(sessions)
        if n == 2:  # duplicated session id inside one haystack (dedup rule)
            ids_h.append(ids[0])
            sess_h.append(sessions[0])
        dates = [f"2023/05/{10 + j:02d} (Wed) 0{j}:15" for j in range(len(ids_h))]
        out.append({
            "question_id": qid, "question_type": qtype,
            "question": f"What is the {topic} value in my {topic} project?", "answer": answer,
            "question_date": "2023/06/01 (Thu) 10:00", "haystack_session_ids": ids_h,
            "haystack_dates": dates, "haystack_sessions": sess_h, "answer_session_ids": gold,
        })  # fmt: skip
    return out


def make_m() -> list[dict]:
    rng = random.Random(11)
    sessions = {f"m{j:03d}": _filler_session(rng, 2 + j % 3) for j in range(12)}
    out = []
    for h in range(4):
        ids = sorted(rng.sample(sorted(sessions), 8))
        out.append({
            "question_id": f"m{h}", "question_type": "multi-session", "question": "q", "answer": "a",
            "question_date": "2023/06/01 (Thu) 10:00", "haystack_session_ids": ids,
            "haystack_dates": [f"2023/04/{1 + int(i[1:])%28:02d} (Sat) 08:00" for i in ids],
            "haystack_sessions": [sessions[i] for i in ids], "answer_session_ids": [ids[0]],
        })  # fmt: skip
    return out


if __name__ == "__main__":
    for name, data in (("lme_s_mini.json", make_s()), ("lme_m_mini.json", make_m())):
        (HERE / name).write_text(
            json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
