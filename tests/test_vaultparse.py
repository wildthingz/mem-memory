"""Pure parsing rules of the importer (architecture-v2 §2.4), no database."""

from mem.vaultparse import normalize, parse_note, scan_links, scan_tags


def parse(text: str, path: str = "personal/Note.md"):
    return parse_note(path, normalize(text))


def test_empty_and_frontmatter_only_notes_have_no_chunks():
    assert parse("").chunks == []
    note = parse("---\nstatus: draft\n---\n")
    assert note.chunks == [] and note.props == [("status", "draft")]


def test_tags_skip_code_hex_colours_numbers_headings_and_anchors():
    text = "# Title\n\n#real #a/b #2026 #ff8800 `#inline` [[X#Anchor]] http://h/#frag\n\n```\n#fenced\n```\n"
    assert scan_tags(text) == ["real", "a/b"]


def test_escaped_pipe_in_a_table_cell_is_the_alias_separator():
    (link,) = scan_links(r"| [[Ana Silva\|Ana]] |")
    assert (link.target, link.alias, link.anchor) == ("Ana Silva", "Ana", None)


def test_link_to_a_heading_of_the_same_note_has_an_empty_target():
    (link,) = scan_links("See [[#Role]].")
    assert (link.target, link.anchor) == ("", "Role")


def test_unquoted_frontmatter_wikilink_is_read_as_a_link_not_a_prop():
    note = parse("---\nowner: [[Ana Silva]]\nkind: [a, b]\n---\nBody\n")
    assert [(f.field, f.rel, f.link.target) for f in note.fm_links] == [
        ("owner", "owner", "Ana Silva")
    ]
    assert note.props == [("kind", "a"), ("kind", "b")]


def test_frontmatter_key_named_like_a_core_relation_gets_a_prefix():
    note = parse('---\nsupersedes: "[[Old]]"\n---\nBody\n')
    assert note.fm_links[0].rel == "fm_supersedes"


def test_oversized_code_block_stays_one_chunk_and_hash_ignores_comments():
    code = "```\n" + "word " * 600 + "\n```"
    note = parse(f"# T\n\n## Big\n\n{code}\n\nTail paragraph.\n")
    assert [c.key for c in note.chunks] == ["T > Big#1/p0", "T > Big#1/p1"]
    assert normalize("a %%x\ny%% b\r\n") == "a  b\n"


def test_heading_inside_a_fence_does_not_split_and_h4_does_not_either():
    note = parse("## A\n\n" + "text " * 30 + "\n\n#### deep\n\n```\n## no\n```\n")
    assert [c.key for c in note.chunks] == ["A#1/p0"]


def test_chunks_reconstruct_the_normalised_body_heading_lines_included():
    text = (
        "# T\n\nIntro line.\n\n## A\n\n"
        + "alpha word " * 60
        + "\n\n### A1\n\nDeep text.\n\n## B\n\nTail.\n"
    )
    note = parse(text)
    got = " ".join("\n".join(c.text for c in note.chunks).split())
    assert got == " ".join(normalize(text).split())
    assert note.chunks[0].text.startswith("# T")
