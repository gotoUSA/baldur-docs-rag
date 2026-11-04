import subprocess

import pytest

from baldur_docs_rag.documents import is_included, load_documents, resolve_commit
from baldur_docs_rag.errors import SourceReadError


def git(repo, *args):
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.com",
            *args,
        ],
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init")
    (tmp_path / "docs" / "sub").mkdir(parents=True)
    (tmp_path / "docs" / "sub" / "a.md").write_text("committed", encoding="utf-8")
    (tmp_path / "README.md").write_text("readme", encoding="utf-8")
    (tmp_path / "CHANGELOG.md").write_text("changes", encoding="utf-8")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-m", "init")
    return tmp_path


@pytest.mark.parametrize(
    "path, expected",
    [
        ("README.md", True),
        ("README.ko.md", True),
        ("ARCHITECTURE.md", True),
        ("docs/concepts/a.md", True),
        ("docs/image.png", False),
        ("CHANGELOG.md", False),
        ("THIRD_PARTY_LICENSES.md", False),
        ("src/docs/a.md", False),
    ],
)
def test_document_selection(path, expected):
    assert is_included(path) is expected


def test_only_selected_files_are_loaded(repo):
    documents = load_documents(repo, resolve_commit(repo, "HEAD"))

    assert [d.path for d in documents] == ["README.md", "docs/sub/a.md"]


def test_content_comes_from_the_commit_not_the_working_tree(repo):
    commit = resolve_commit(repo, "HEAD")
    (repo / "docs" / "sub" / "a.md").write_text("uncommitted edit", encoding="utf-8")

    by_path = {d.path: d.text for d in load_documents(repo, commit)}

    assert by_path["docs/sub/a.md"] == "committed"


def test_commit_with_no_documents_fails_instead_of_passing_empty(tmp_path):
    git(tmp_path, "init")
    (tmp_path / "other.txt").write_text("x", encoding="utf-8")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-m", "init")

    with pytest.raises(SourceReadError, match="No documents"):
        load_documents(tmp_path, resolve_commit(tmp_path, "HEAD"))


def test_missing_repository_fails(tmp_path):
    with pytest.raises(SourceReadError):
        resolve_commit(tmp_path / "missing", "HEAD")


def test_non_utf8_document_fails(repo):
    (repo / "docs" / "bad.md").write_bytes(b"\xff\xfe\x00bad")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "bad")

    with pytest.raises(SourceReadError, match="not UTF-8"):
        load_documents(repo, resolve_commit(repo, "HEAD"))
