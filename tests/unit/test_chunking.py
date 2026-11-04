from baldur_docs_rag.chunking import split_markdown

MAX = 200


def split(text, max_chars=MAX):
    return split_markdown("docs/a.md", text, max_chars)


def test_heading_path_follows_nesting_and_resets_on_a_higher_level():
    text = "# A\nintro\n## B\nb text\n### C\nc text\n## D\nd text\n"

    assert [c.heading for c in split(text)] == ["A", "A > B", "A > B > C", "A > D"]


def test_comment_inside_code_fence_is_not_a_heading():
    text = "# A\n```bash\n# install it\npip install baldur\n```\n"

    chunks = split(text)

    assert [c.heading for c in chunks] == ["A"]
    assert "# install it" in chunks[0].content


def test_heading_with_no_body_produces_no_chunk():
    assert [c.heading for c in split("# A\n\n## B\nb text\n")] == ["A > B"]


def test_yaml_front_matter_is_removed():
    chunks = split("---\ntitle: Secret title\ndescription: x\n---\n# A\nbody\n")

    assert "Secret title" not in chunks[0].content


def test_leading_horizontal_rule_is_not_treated_as_front_matter():
    chunks = split("---\nplain text\n\n---\nmore\n")

    assert "plain text" in chunks[0].content


def test_text_before_the_first_heading_has_an_empty_heading():
    assert split("preamble\n# A\nbody\n")[0].heading == ""


def test_chunk_indexes_are_sequential_within_a_file():
    text = "# A\n" + "\n\n".join(f"paragraph {i} " * 8 for i in range(20))

    chunks = split(text)

    assert len(chunks) > 1
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_oversized_section_keeps_the_heading_path_on_every_piece():
    text = "# A\n## B\n" + "\n\n".join(f"paragraph {i} " * 8 for i in range(20))

    assert {c.heading for c in split(text)} == {"A > B"}


def test_no_chunk_exceeds_the_limit_for_any_oversized_shape():
    long_line = "x" * (MAX * 3 + 7)
    long_fence = "```\n" + "\n".join(f"line {i}" for i in range(200)) + "\n```"
    paragraphs = "\n\n".join(f"paragraph {i} " * 8 for i in range(20))
    text = f"# A\n{long_line}\n\n{long_fence}\n\n{paragraphs}\n"

    assert all(len(c.content) <= MAX for c in split(text))


def test_every_body_line_survives_splitting():
    source = [f"line number {i}" for i in range(300)]
    text = "# A\n" + "\n".join(source) + "\n\n" + "\n\n".join(source[:50])

    joined = "\n".join(c.content for c in split(text))

    assert all(line in joined for line in source)


def test_short_fence_is_not_split_across_chunks():
    filler = "\n\n".join(f"paragraph {i} " * 8 for i in range(6))
    fence = "```python\nfirst()\n\nsecond()\n```"

    chunks = split(f"# A\n{filler}\n\n{fence}\n\n{filler}\n")

    assert any(fence in c.content for c in chunks)


def test_empty_text_produces_no_chunks():
    assert split("") == []


def test_windows_line_endings_are_handled():
    assert [c.heading for c in split("# A\r\nbody\r\n## B\r\nmore\r\n")] == [
        "A",
        "A > B",
    ]


def test_oversized_section_produces_no_heading_only_chunk():
    # Real case: docs/concepts/pro/release-notes.md produced a chunk of only "### Added",
    # because the block after the heading was too large to join it.
    big_block = "\n".join(f"- change number {i}" for i in range(40))

    chunks = split(f"# A\n### Added\n\n{big_block}\n")

    assert all(c.content.strip() != "### Added" for c in chunks)
    assert {c.heading for c in chunks} == {"A > Added"}
