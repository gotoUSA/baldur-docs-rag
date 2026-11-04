"""Split Markdown into chunks along headings, then blank lines, then lines."""

import re
from dataclasses import dataclass

HEADING_PATH_SEPARATOR = " > "
MAX_SPLIT_HEADING_LEVEL = 3

_FRONT_MATTER = re.compile(r"\A---\n[A-Za-z_-]+:.*?\n---\n", re.DOTALL)
_HEADING = re.compile(r"^(#{1,%d})\s+(.+?)\s*#*\s*$" % MAX_SPLIT_HEADING_LEVEL)
_HEADING_ONLY = re.compile(r"#{1,6}\s+[^\n]+")
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")


@dataclass(frozen=True)
class Chunk:
    source_path: str
    chunk_index: int
    heading: str
    content: str


class _FenceTracker:
    """Follows code fences, so a '# comment' in a code block is not a heading."""

    def __init__(self) -> None:
        self._marker: str | None = None

    @property
    def inside(self) -> bool:
        return self._marker is not None

    def feed(self, line: str) -> None:
        match = _FENCE.match(line)
        if match is None:
            return
        marker, rest = match.groups()
        if self._marker is None:
            self._marker = marker
        elif (
            marker[0] == self._marker[0]
            and len(marker) >= len(self._marker)
            and not rest.strip()
        ):
            self._marker = None


def split_markdown(source_path: str, text: str, max_chars: int) -> list[Chunk]:
    normalized = text.replace("\r\n", "\n")
    body = _FRONT_MATTER.sub("", normalized, count=1)

    chunks: list[Chunk] = []
    for heading, section in _split_sections(body):
        for piece in _split_oversized(section, max_chars):
            chunks.append(
                Chunk(
                    source_path=source_path,
                    chunk_index=len(chunks),
                    heading=heading,
                    content=piece,
                )
            )
    return chunks


def _split_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    heading_path = ""
    lines: list[str] = []
    has_body = False
    fence = _FenceTracker()

    def flush() -> None:
        # A heading directly followed by another heading has nothing to retrieve.
        if has_body:
            sections.append((heading_path, "\n".join(lines)))

    for line in text.split("\n"):
        was_inside = fence.inside
        fence.feed(line)
        match = None if was_inside or fence.inside else _HEADING.match(line)
        if match is None:
            lines.append(line)
            has_body = has_body or bool(line.strip())
            continue

        flush()
        level = len(match.group(1))
        stack = [item for item in stack if item[0] < level]
        stack.append((level, match.group(2)))
        heading_path = HEADING_PATH_SEPARATOR.join(title for _, title in stack)
        lines = [line]
        has_body = False

    flush()
    return sections


def _split_oversized(section: str, max_chars: int) -> list[str]:
    section = section.strip()
    if len(section) <= max_chars:
        return [section]

    pieces: list[str] = []
    for block in _split_blocks(section):
        if len(block) <= max_chars:
            pieces.append(block)
        else:
            pieces.extend(_pack(_split_long_lines(block, max_chars), max_chars, "\n"))
    packed = [piece.strip() for piece in _pack(pieces, max_chars, "\n\n")]
    # The heading stays in Chunk.heading, so a piece holding only the heading adds nothing.
    return [piece for piece in packed if not _HEADING_ONLY.fullmatch(piece)]


def _split_blocks(text: str) -> list[str]:
    """Split on blank lines, keeping each code fence in one block."""
    blocks: list[str] = []
    current: list[str] = []
    fence = _FenceTracker()

    for line in text.split("\n"):
        fence.feed(line)
        if line.strip() or fence.inside:
            current.append(line)
        elif current:
            blocks.append("\n".join(current))
            current = []
    if current:
        blocks.append("\n".join(current))
    return blocks


def _split_long_lines(block: str, max_chars: int) -> list[str]:
    parts: list[str] = []
    for line in block.split("\n"):
        parts.extend(
            line[i : i + max_chars] for i in range(0, max(len(line), 1), max_chars)
        )
    return parts


def _pack(parts: list[str], max_chars: int, separator: str) -> list[str]:
    packed: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current}{separator}{part}" if current else part
        if len(candidate) <= max_chars:
            current = candidate
        else:
            packed.append(current)
            current = part
    if current:
        packed.append(current)
    return packed
