"""Versioned data of SYN (benchmarks-v2 §3.2): relations, cue lists, negations, filler, name and
value lists. Changing anything here changes the generated worlds: bump `SYN_VERSION` and the golden
file (`bench/synth/golden/seed1_S.sha256`), and `bench_version` if a recorded result is affected.

Relation cue lists are DISJOINT across relations: no cue of one relation equals, contains or is
contained in a cue of another (tests/bench_t0/test_syn_data.py). Filler sentences contain no cue,
no negation phrase and no slot value. All cue and filler text is lower-case ASCII apart from the
subject and value strings the generator inserts.
"""

from __future__ import annotations

from dataclasses import dataclass

SYN_VERSION = 3  # 3: an update's gold counterpart is the member valid at its as_of (MEM2-156)
TEMPLATE_VERSION = 1
CUES_VERSION = 1
TZ = "Europe/Amsterdam"  # the configured zone date-only `as_of` values are read in
WORLD_START = "2024-11-01"  # day 0; ops run over 730 days
WORLD_DAYS = 730


@dataclass(frozen=True)
class Relation:
    name: str
    subject: str  # entity type of the subject
    multi: bool  # several current values allowed (compatible_add targets these)
    mem_type: str  # memory `type` written
    cues: tuple[str, ...]
    values: str  # key into VALUES
    question: str  # {s} = subject name


RELATIONS: dict[str, Relation] = {r.name: r for r in (
    Relation("lives_in", "person", False, "fact",
             ("lives in", "lived in", "living in", "based in", "moved to", "home is",
              "resident of"), "city", "Where does {s} live?"),
    Relation("works_at", "person", False, "fact",
             ("works at", "worked at", "working at", "employed by", "joined"), "company",
             "Where does {s} work?"),
    Relation("has_role", "person", False, "fact",
             ("works as", "role is", "serves as", "promoted to", "title is"), "job",
             "What is the job role of {s}?"),
    Relation("favourite_food", "person", False, "preference",
             ("favourite food is", "favorite food is", "loves eating", "most loved dish is"),
             "food", "What is the favourite food of {s}?"),
    Relation("pet", "person", False, "fact",
             ("has a pet named", "pet is called", "owns a pet called", "adopted a pet named"),
             "pet", "What is the name of the pet of {s}?"),
    Relation("speaks", "person", True, "fact",
             ("speaks", "fluent in", "is learning"), "language",
             "Which languages does {s} speak?"),
    Relation("hobby", "person", False, "preference",
             ("enjoys", "hobby is", "spends weekends on"), "hobby",
             "What is the hobby of {s}?"),
    Relation("car", "person", False, "fact",
             ("drives a", "car is a"), "car", "What does {s} drive?"),
    Relation("commute", "person", False, "fact",
             ("commutes by", "travels to work by"), "commute", "How does {s} commute?"),
    Relation("reads", "person", True, "preference",
             ("is reading", "recently read"), "book", "What is {s} reading?"),
    Relation("codename", "project", False, "fact",
             ("codename is", "code named", "internally called"), "codename",
             "What is the codename of {s}?"),
    Relation("venue", "project", False, "event",
             ("launches at", "launch venue is", "will launch at"), "venue",
             "Where does {s} launch?"),
    Relation("uses_tool", "project", True, "fact",
             ("uses", "built with", "runs on"), "tool", "What does {s} use?"),
    Relation("sponsor", "project", False, "fact",
             ("sponsored by", "sponsor is"), "sponsor", "Who sponsors {s}?"),
    Relation("hq", "org", False, "fact",
             ("headquartered in", "hq is in", "main office is in"), "hq",
             "Where is {s} headquartered?"),
    Relation("mascot", "org", False, "fact",
             ("mascot is", "mascot is called"), "mascot", "What is the mascot of {s}?"),
    Relation("founder", "org", False, "fact",
             ("founded by", "founder is"), "founder", "Who founded {s}?"),
    Relation("industry", "org", False, "fact",
             ("operates in the", "industry is"), "industry", "What industry is {s} in?"),
    Relation("known_for", "place", False, "fact",
             ("is known for", "famous for", "best known for"), "famous", "What is {s} known for?"),
    Relation("landmark", "place", False, "fact",
             ("landmark is", "main landmark is"), "landmark", "What is the landmark of {s}?"),
    Relation("festival", "place", False, "event",
             ("hosts the", "annual festival is"), "festival", "What festival does {s} host?"),
)}  # fmt: skip

# Negation phrases of the B-W5 alignment rule 4 (benchmarks-v2 B-W5), versioned.
NEGATIONS: tuple[str, ...] = (
    "not", "n't", "never", "no", "no longer", "not anymore", "any more", "former", "formerly",
    "ex", "used to", "previously", "until", "moved away", "moved out of", "left", "stopped", "quit",
)  # fmt: skip

# Update cues (B-W3: half of the updates carry one). A prefix; the sentence keeps its relation cue.
UPDATE_PREFIXES: tuple[str, ...] = ("Update:", "Now", "As of now,", "Correction:", "Latest:")

FILLER: tuple[str, ...] = (
    "Noted during the weekly sync.", "Mentioned in passing yesterday.",
    "Worth remembering for later.", "Shared over coffee.", "It came up in conversation.",
    "Confirmed by email this week.", "Written down after the call.", "Said with a smile.",
    "Raised near the end of the meeting.", "Added to the running notes.",
)  # fmt: skip

FIRST_NAMES: tuple[str, ...] = (
    "Ana", "Bram", "Clara", "Dmitri", "Elena", "Farid", "Greta", "Hugo", "Ingrid", "Jonas",
    "Katya", "Leon", "Marta", "Nikolai", "Olga", "Pavel", "Quinn", "Rosa", "Sven", "Tamar",
    "Ulrich", "Vera", "Willem", "Xenia", "Yusuf", "Zelda", "Anders", "Beatriz", "Carlos", "Dagny",
    "Emil", "Fiona", "Gideon", "Hanna", "Ivor", "Julia", "Konrad", "Lucia", "Mateo", "Nadia",
    "Oskar", "Petra", "Rafael", "Selma", "Teodor", "Ursula", "Viktor", "Wanda", "Yara", "Zoran",
    "Astrid", "Bastian", "Cecile", "Dario", "Esther", "Felix", "Gianna", "Henrik", "Irina",
    "Jorge", "Kirsten", "Lars", "Mirela", "Nils", "Odile", "Piet", "Ramona", "Stefan", "Thea",
    "Ugo", "Valeria", "Wim", "Ximena", "Yannick", "Zora", "Agnes", "Boris", "Cora", "Dorian",
    "Edda", "Fabian", "Gwen", "Holger", "Iris", "Joris", "Kaya", "Lennart", "Mona", "Nico",
)  # fmt: skip

LAST_NAMES: tuple[str, ...] = (
    "Silva", "Lopez", "Hendrik", "Novak", "Moreau", "Kowalski", "Bianchi", "Johansson",
    "Petrov", "Okafor", "Tanaka", "Haddad", "Verhoeven", "Castillo", "Lindqvist", "Marchetti",
    "Dubois", "Abramov", "Sorensen", "Fontana", "Brandt", "Costa", "Eriksen", "Fischer",
    "Guerrero", "Hofmann", "Ivanova", "Jankowski", "Keller", "Laurent", "Mendes", "Neumann",
    "Ortega", "Pereira", "Quist", "Romano", "Schmidt", "Thorsen", "Urbina", "Vasquez",
    "Wagner", "Yilmaz", "Zimmer", "Aalto", "Bakker", "Cardoso", "Dragan", "Engel", "Franke",
    "Gruber", "Holm", "Imhof", "Jansen", "Krause", "Larsen", "Madsen", "Nygaard", "Olsen",
    "Prins", "Rinaldi", "Stoica", "Tamm", "Uzun", "Vogel", "Weiss", "Yamada", "Zeman",
)  # fmt: skip

THING_NAMES: tuple[str, ...] = (
    "Atlas", "Kestrel", "Meridian", "Orion", "Vega", "Helix", "Nimbus", "Quartz", "Solstice",
    "Tundra", "Aurora", "Basalt", "Cobalt", "Delta", "Ember", "Fjord", "Granite", "Harbor",
    "Indigo", "Juniper", "Krypton", "Lagoon", "Monsoon", "Nebula", "Onyx", "Pinnacle", "Quasar",
    "Raven", "Saffron", "Topaz", "Umbra", "Vortex", "Willow", "Zephyr", "Alpine", "Beacon",
    "Cascade", "Dynamo", "Equinox", "Falcon", "Glacier", "Horizon", "Ivory", "Jasper",
)  # fmt: skip

NICKNAMES: tuple[str, ...] = (
    "Annie", "Bobby", "Chip", "Dizzy", "Eddie", "Flip", "Gus", "Hank", "Izzy", "Jinx",
    "Kit", "Lulu", "Moxie", "Nugget", "Ozzie", "Pip", "Quincy", "Rusty", "Sparky", "Tink",
    "Ulla", "Vinnie", "Wiggy", "Yogi", "Zippy", "Biscuit", "Cricket", "Doodle", "Fizz", "Gadget",
)  # fmt: skip

# Value lists. Each value is ONE lower-case-distinct token; the generator draws without replacement
# across the whole world and invents pronounceable words when a list runs out.
VALUES: dict[str, tuple[str, ...]] = {
    "city": ("Lisbon", "Oslo", "Kraków", "Porto", "Gdansk", "Bergen", "Ghent", "Leuven",
             "Trieste", "Valencia", "Malmo", "Turku", "Brno", "Graz", "Split", "Bruges",
             "Cork", "Aarhus", "Lyon", "Genoa", "Seville", "Tartu", "Riga", "Plovdiv"),
    "company": ("Brightwell", "Corvane", "Dunmore", "Elmstead", "Fairhaven", "Grayling",
                "Hartwood", "Ironvale", "Jadeport", "Kelmora", "Lanterne", "Mosswick",
                "Northgate", "Oakridge", "Pinecroft", "Quillon", "Redmarsh", "Stonebridge",
                "Thornfield", "Umberly"),
    "job": ("analyst", "engineer", "designer", "architect", "auditor", "curator", "dispatcher",
            "editor", "forester", "gaffer", "hydrologist", "inspector", "joiner", "kilnmaster",
            "locksmith", "mechanic", "navigator", "optician", "plumber", "rigger"),
    "food": ("risotto", "pierogi", "falafel", "paella", "goulash", "ramen", "tagine", "ceviche",
             "gnocchi", "borscht", "moussaka", "pho", "biryani", "pavlova", "empanada", "laksa"),
    "pet": ("Pretzel", "Mochi", "Pepper", "Waffles", "Noodle", "Pickle", "Bean", "Turnip",
            "Maple", "Ziggy", "Fudge", "Clover", "Peanut", "Truffle", "Marble", "Sprout"),
    "language": ("Portuguese", "Polish", "Norwegian", "Dutch", "Finnish", "Czech", "Croatian",
                 "Catalan", "Estonian", "Latvian", "Slovene", "Basque", "Maltese", "Icelandic",
                 "Galician", "Romansh"),
    "codename": ("Falconet", "Gimlet", "Harrier", "Ibex", "Jackal", "Kodiak", "Lynx", "Marten",
                 "Narwhal", "Osprey", "Puffin", "Quokka", "Reindeer", "Stoat", "Tapir", "Vole"),
    "venue": ("Arenanord", "Bellhall", "Cityforum", "Dockyard", "Eastpier", "Foundry",
              "Greenstage", "Hangarnine", "Ironworks", "Jettyhall", "Kilnhouse",
              "Lightstage"),
    "tool": ("Postgres", "Rustc", "Svelte", "Bazel", "Terraform", "Kafka", "Nginx", "Redis",
             "Grafana", "Ansible", "Prometheus", "Istio", "Vagrant", "Jenkins", "Consul", "Nomad"),
    "hq": ("Rotterdam", "Utrecht", "Ljubljana", "Bratislava", "Vilnius", "Zagreb", "Salzburg",
           "Antwerp", "Bologna", "Dresden", "Linz", "Maribor"),
    "mascot": ("Otterly", "Badgerton", "Hedgie", "Marmot", "Newtie", "Penguino", "Squirrelly",
               "Toadsworth", "Weaselby", "Yakety"),
    "hobby": ("pottery", "kayaking", "birdwatching", "chess", "gardening", "origami",
              "fencing", "bouldering", "calligraphy", "beekeeping", "orienteering", "knitting"),
    "car": ("Volvo", "Skoda", "Peugeot", "Fiat", "Saab", "Seat", "Dacia", "Lancia", "Citroen",
            "Renault", "Opel", "Mini"),
    "commute": ("tram", "ferry", "scooter", "funicular", "bicycle", "tandem", "skates", "kayak",
                "rickshaw", "monorail"),
    "book": ("Middlemarch", "Ulysses", "Dune", "Emma", "Kokoro", "Persuasion", "Solaris",
             "Beloved", "Rebecca", "Siddhartha", "Germinal", "Hunger"),
    "sponsor": ("Ardent", "Beaconfield", "Cravenmoor", "Dovetail", "Eastvale", "Foxglove",
                "Goldcrest", "Hawkridge", "Inglenook", "Jaybird"),
    "founder": ("Ashworth", "Blackwood", "Carrow", "Dunstan", "Everly", "Fairbanks", "Gallagher",
                "Hollis", "Ingram", "Jessop"),
    "industry": ("shipping", "textiles", "forestry", "brewing", "logistics", "fisheries",
                 "mining", "ceramics", "publishing", "printing"),
    "landmark": ("Bellfry", "Citadel", "Obelisk", "Rotunda", "Ziggurat", "Aqueduct", "Pagoda",
                 "Minaret", "Lookout", "Gatehouse"),
    "festival": ("Lantern", "Harvest", "Midsummer", "Carnival", "Mistral", "Kite", "Regatta",
                 "Folklore", "Jazzfest", "Cider"),
    "famous": ("cheese", "ferries", "lacework", "pottery", "glassblowing", "marzipan", "tulips",
               "violins", "windmills", "woodcarving", "clockmaking", "saffron"),
}  # fmt: skip

SYLLABLES: tuple[str, ...] = (
    "ba", "be", "bi", "bo", "da", "de", "di", "do", "fa", "fe", "fi", "ga", "ge", "gi", "ka",
    "ke", "ki", "ko", "la", "le", "li", "lo", "ma", "me", "mi", "mo", "na", "ne", "ni", "no",
    "pa", "pe", "pi", "po", "ra", "re", "ri", "ro", "sa", "se", "si", "so", "ta", "te", "ti",
    "to", "va", "ve", "vi", "vo", "za", "ze", "zi", "zo",
)  # fmt: skip

SIZES: dict[str, dict[str, int]] = {
    "S": {"people": 20, "projects": 8, "orgs": 4, "places": 5, "first_collisions": 3,
          "cross_kind": 1, "ops": 200},
    "M": {"people": 60, "projects": 25, "orgs": 10, "places": 15, "first_collisions": 8,
          "cross_kind": 3, "ops": 1000},
    "L": {"people": 400, "projects": 150, "orgs": 60, "places": 60, "first_collisions": 40,
          "cross_kind": 10, "ops": 10000},
}  # fmt: skip

# Op shares (benchmarks-v2 §3.2). Quotas are floor(share * ops); the rest goes to new_fact.
SHARES: dict[str, float] = {
    "new_fact": 0.52, "update": 0.15, "dup_exact": 0.08, "dup_paraphrase": 0.12,
    "compatible_add": 0.05, "ambiguous_mention": 0.05, "retract": 0.03,
}  # fmt: skip
UPDATE_SHARES: dict[str, float] = {
    "value_returns": 0.20, "future": 0.10, "late": 0.05, "same_instant": 0.05,
}  # fmt: skip
TYPO_RATE = 0.03

DEV_SEEDS = (1, 2, 3, 4, 5)
HELD_OUT_SEEDS = (101, 102, 103, 104, 105)


def all_cues() -> dict[str, tuple[str, ...]]:
    return {r.name: r.cues for r in RELATIONS.values()}


def cue_conflicts() -> list[tuple[str, str, str, str]]:
    """(rel_a, cue_a, rel_b, cue_b) for every pair of cues of different relations that are equal or
    one contains the other. Empty = the lists are disjoint."""
    bad = []
    items = [(r.name, c) for r in RELATIONS.values() for c in r.cues]
    for i, (ra, ca) in enumerate(items):
        for rb, cb in items[i + 1 :]:
            if ra != rb and (ca in cb or cb in ca):
                bad.append((ra, ca, rb, cb))
    return bad
