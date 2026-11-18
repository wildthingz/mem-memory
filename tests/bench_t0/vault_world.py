"""Vault worlds for the P4 T0 tests.

`MINI` is a cut of the frozen mini-vault of the fixture task (tests/fixtures/minivault on
bosmang/p4-fixture; notes copied verbatim) plus the files the importer skips; `make_mini` writes it.
`make_world` generates a vault big enough for the PG quotas: notes with unique vocabulary (so the
stub embedder and BM25 find them), wikilinks between neighbours (linked pairs), a `projects/client-x`
folder that the routes send to the work store.
"""

from __future__ import annotations

import os
from pathlib import Path

MTIME = 1768478400.0  # 2026-01-15T12:00:00Z
ROUTES = {"projects/client-x": "work"}

MINI: dict[str, str] = {
    "personal/2026-04-10_Meeting with Ana.md": '---\ncreated: 2026-04-10\naliases: [Ana sync, Weekly Ana]\ntags: [meeting]\nattendees: ["[[Ana Silva]]", "[[Bruno]]"]\nproject: "[[Atlas]]"\nstatus: draft\nroom: B12\n---\n# Ana sync notes\n\n## Agenda\n\nWe met [[Ana Silva]] to review the Atlas rollout. %%private reminder: do not share the budget number%% Later [[Ana Silva|Ana]] summarised the open risks, and the section [[Ana Silva#Role]] describes who owns what. The block [[Ana Silva#^risk1]] lists the first risk, and the unresolved [[Ghost Topic]] still needs a page of its own.\n\n> [!warning] Deadline\n> The review is due before the end of the quarter, so the open items need owners this week.\n\n## Decisions\n\nDecided to keep the weekly cadence and to move the status report to Fridays. The diagram ![[diagram.png]] shows the new flow, and the whiteboard is kept in [[board.canvas]]. Follow-up with [[Bruno]] on staffing, plus a #meeting/weekly reminder for the next review. The follow-up items are tracked in the shared log, reviewed every week, and closed only when the owner confirms the result in writing.\n',
    "personal/people/Ana Silva.md": '---\ntags: [person]\naliases: [Ana, A. Silva]\nreports_to: "[[Carla Diaz]]"\nrole: engineer\n---\n# Ana Silva\n\n## Role\n\nAna is the lead engineer on the Atlas project and reports to Carla, with a focus on retrieval quality, data import and the review of every change before it lands in the main branch. The follow-up items are tracked in the shared log, reviewed every week, and closed only when the owner confirms the result in writing.\n## Contact\n\nAna works from the home office on Tuesdays and Thursdays and from the main office on the other days, and answers messages within a few hours. The follow-up items are tracked in the shared log, reviewed every week, and closed only when the owner confirms the result in writing. The follow-up items are tracked in the shared log, reviewed every week, and closed only when the owner confirms the result in writing.\n',
    "personal/2026-03-02_Trip planning.md": "Spring trip: #travel and #trip/2026.\n\n## Packing\n\nPack light: two pairs of trousers, three shirts, a rain jacket and the charger bundle. Keep the passport and the printed tickets in the front pocket of the carry-on, and put anything liquid in a clear bag before the airport check. The follow-up items are tracked in the shared log, reviewed every week, and closed only when the owner confirms the result in writing.\n\n## Packing\n\nSecond pass on the bag the night before leaving: weigh it, remove one shirt if it is over the limit, and check that the power adapter and the spare memory card are inside. The colour of the tag is #ff8800 and the code sample `#notatag` is not a tag.\n\n## Food\n\nBook the dinner place near the harbour for the first night, and look for a bakery close to the hotel for breakfast. Check opening hours, because many small places close on Mondays and take their summer break early. The follow-up items are tracked in the shared log, reviewed every week, and closed only when the owner confirms the result in writing.\n\n## Notes\n\nRemember the umbrella.\n\n## Budget\n\nSet a daily budget for food, transport and tickets, keep the receipts in the shared folder, and compare the total with the plan when the trip is over so the next trip can be planned with better numbers. The follow-up items are tracked in the shared log, reviewed every week, and closed only when the owner confirms the result in writing.\n\n```\n# not a heading and #nottag in a code block\n```\n",
    "projects/Atlas.md": '---\ntags: [project]\nowner: "[[Ana Silva]]"\nstage: rollout\n---\n# Atlas\n\n## Goal\n\nAtlas is the rollout of the new search stack: replace the old index, move every team to the new query service, and keep the quality numbers above the agreed floor during the whole migration. The follow-up items are tracked in the shared log, reviewed every week, and closed only when the owner confirms the result in writing.\n## Risks\n\nThe main risks are the data migration window, the capacity of the query service, and the training of the support team before the cut-over date. The follow-up items are tracked in the shared log, reviewed every week, and closed only when the owner confirms the result in writing. The follow-up items are tracked in the shared log, reviewed every week, and closed only when the owner confirms the result in writing.\n',
    "work/Platform.md": "---\ntags: [team]\naliases: [Platform team]\n---\n# Platform\n\n## Mission\n\nThe platform team runs the shared services, owns the on-call rota and the deployment tooling, and supports every product team that needs infrastructure advice or a review of a design. The follow-up items are tracked in the shared log, reviewed every week, and closed only when the owner confirms the result in writing. The follow-up items are tracked in the shared log, reviewed every week, and closed only when the owner confirms the result in writing.\n",
    "personal/people/Carla Diaz.md": "---\ntags: [person]\n---\n# Carla Diaz\n\n## Role\n\nCarla is the engineering director who owns the budget, the hiring plan and the yearly roadmap, and she meets each team lead once a week to hear about risks and progress. The follow-up items are tracked in the shared log, reviewed every week, and closed only when the owner confirms the result in writing. The follow-up items are tracked in the shared log, reviewed every week, and closed only when the owner confirms the result in writing.\n",
    "work/2026-05-01_Standup.md": "Standup with the [[Platform]] team: the deploy pipeline is green, the on-call handover is done, and the next review of the capacity plan is scheduled for Thursday afternoon with everyone present. The follow-up items are tracked in the shared log, reviewed every week, and closed only when the owner confirms the result in writing. The follow-up items are tracked in the shared log, reviewed every week, and closed only when the owner confirms the result in writing. The follow-up items are tracked in the shared log, reviewed every week, and closed only when the owner confirms the result in writing.\n",
    "projects/client-x/Kickoff.md": "---\ncreated: 2026-05-02\ntags: [meeting]\n---\n# Client X kickoff\n\n## Scope\n\nThe folder projects/client-x is mapped to the work store by the config override, so this note proves that an override wins over the default routing of the projects folder to the personal store. The follow-up items are tracked in the shared log, reviewed every week, and closed only when the owner confirms the result in writing.\n",
    "personal/diagram.png": "png\n",
}

SKIPPED = {
    ".obsidian/app.json": "{}\n",
    "CLAUDE.md": "# skip\n",
    "Templates/Daily.md": "# Daily\n",
    "export/personal/trip.md": "# rendered copy\n\nrendered copy of trip.\n",
}
LIFECYCLE = r"""
vault: tests/fixtures/minivault/vault
routes: {projects/client-x: work}
edit:
  path: personal/people/Ana Silva.md
  find: from the home office on Tuesdays
  replace: from the home office on Mondays
  chunk_key: "Ana Silva > Contact#1/p0"
add:
  path: personal/2026-10-23_New idea.md
  text: |
    # New idea

    ## Sketch

    A new idea about the quarterly planning cycle that has enough words in it to stand on its own as a section of the new note, so that the importer writes exactly one new chunk for it and nothing else.
rename:
  from: personal/2026-03-02_Trip planning.md
  to: personal/2026-03-02_Trip planning renamed.md
delete:
  path: projects/Atlas.md
remove:
  path: personal/2026-04-10_Meeting with Ana.md
  alias: Weekly Ana
  alias_edit: {find: "aliases: [Ana sync, Weekly Ana]", replace: "aliases: [Ana sync]"}
  tag: meeting
  tag_edit: {find: "tags: [meeting]", replace: "tags: []"}
  prop_key: room
  prop_edit: {find: "room: B12\n", replace: ""}
  link_rel: attendee
  link_target: personal/people/Ana Silva.md
  link_edit: {find: 'attendees: ["[[Ana Silva]]", "[[Bruno]]"]', replace: 'attendees: ["[[Bruno]]"]'}
insert_section:
  path: projects/Atlas.md
  before: "## Risks"
  text: |
    ## Timeline

    The rollout runs in three waves over the quarter: the pilot teams first, then the product teams that depend on search, and last the support organisation, with a review gate at the end of each wave before the next one may start.
"""


def make_mini(root: Path) -> Path:
    for rel, text in {**MINI, **SKIPPED}.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        os.utime(p, (MTIME, MTIME))
    return root


FOLDERS = {"personal": 14, "projects": 36, "work": 22}
CLIENT_X_FROM = 30  # projects notes from this index on sit in projects/client-x (routed to work)
WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "theta", "iota", "kappa", "lambda", "sigma", "omega", "harbor", "meadow", "quartz", "timber", "violet", "copper", "ember", "glacier"]  # fmt: skip


def note_name(folder: str, i: int) -> str:
    return f"{folder.title()} note {i:02d}"


def note_text(folder: str, i: int, n: int) -> str:
    topic = f"{WORDS[i % len(WORDS)]}{folder[:2]}{i}"
    nxt = note_name(folder, (i + 1) % n)
    return (
        f"---\ncreated: 2026-0{1 + i % 9}-1{i % 9}\naliases: [{topic} desk]\n---\n"
        f"# {note_name(folder, i)}\n\n## Facts\n\n"
        f"The {topic} programme has a budget of {1000 + 7 * i} euro, the lead is Person{folder[:2]}{i}, "
        f"and the review meeting happens every Thursday in the {WORDS[(i + 3) % len(WORDS)]} room "
        f"because the quarterly numbers for {topic} must be checked before anything ships, and the "
        f"minutes of every meeting are stored in the shared folder next to the signed approvals.\n\n"
        f"## History\n\nBefore the budget was {900 + 7 * i} euro. See also [[{nxt}]] for the "
        f"neighbouring {topic} work and the long running notes about the {topic} programme history.\n"
    )


def make_world(root: Path, folders: dict[str, int] | None = None) -> Path:
    for folder, n in (folders or FOLDERS).items():
        for i in range(n):
            rel = f"{folder}/{note_name(folder, i)}.md"
            if (
                folder == "projects" and i >= CLIENT_X_FROM
            ):  # the override sends these to the work store
                rel = f"projects/client-x/{note_name(folder, i)}.md"
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(note_text(folder, i, n), encoding="utf-8")
            os.utime(p, (MTIME, MTIME))
    return root
