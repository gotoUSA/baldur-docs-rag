"""Read the Baldur documents from one git commit, never from the working tree."""

import subprocess
from dataclasses import dataclass
from pathlib import Path

from baldur_docs_rag.errors import SourceReadError

GIT_TIMEOUT_SECONDS = 30
DOCS_PREFIX = "docs/"
DOC_SUFFIX = ".md"
ROOT_DOCUMENTS = frozenset({"README.md", "README.ko.md", "ARCHITECTURE.md"})


@dataclass(frozen=True)
class Document:
    path: str
    text: str


def _git(repo: Path, *args: str) -> bytes:
    command = ["git", "-C", str(repo), *args]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=True,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as e:
        raise SourceReadError(f"git {args[0]} failed in {repo}") from e
    return result.stdout


def resolve_commit(repo: Path, ref: str) -> str:
    return (
        _git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}").decode("ascii").strip()
    )


def is_included(path: str) -> bool:
    return path in ROOT_DOCUMENTS or (
        path.startswith(DOCS_PREFIX) and path.endswith(DOC_SUFFIX)
    )


def load_documents(repo: Path, commit: str) -> list[Document]:
    listing = _git(repo, "ls-tree", "-r", "-z", "--name-only", commit)
    paths = sorted(
        p for p in listing.decode("utf-8").split("\0") if p and is_included(p)
    )
    if not paths:
        raise SourceReadError(f"No documents found at commit {commit} in {repo}")

    documents = []
    for path in paths:
        content = _git(repo, "show", f"{commit}:{path}")
        try:
            documents.append(Document(path=path, text=content.decode("utf-8")))
        except UnicodeDecodeError as e:
            raise SourceReadError(f"{path} at commit {commit} is not UTF-8") from e
    return documents
