"""The real SDK against a local server, for failures that cannot be asked of the API.

An evaluation run hung for 23 minutes on one question although a 120 second timeout
was set. These tests hold GeminiAnswerer to its timeout for each way a server can stall.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from baldur_docs_rag.answer import GeminiAnswerer
from baldur_docs_rag.config import Settings
from baldur_docs_rag.errors import AnswerError, AnswerQuotaError
from baldur_docs_rag.store import SearchHit

HITS = [SearchHit("docs/a.md", 0, "A", "body", 0.7)]
TIMEOUT_SECONDS = 1.0
# Well above the timeout, but short enough that a failing test ends on its own.
STALL_SECONDS = 8
ALLOWED_SECONDS = 4.0


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    behavior = None
    requests_seen = 0


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.server.requests_seen += 1
        self.rfile.read(int(self.headers.get("content-length", 0)))
        self.server.behavior(self)

    def log_message(self, *_args):
        pass


def _rate_limited_with_long_retry_after(handler):
    body = json.dumps({"error": {"code": 429, "message": "slow down"}}).encode()
    handler.send_response(429)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Retry-After", str(STALL_SECONDS))
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _never_replies(_handler):
    time.sleep(STALL_SECONDS)


def _trickles_bytes(handler):
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", "1000")
    handler.end_headers()
    for _ in range(STALL_SECONDS * 4):
        try:
            handler.wfile.write(b" ")
            handler.wfile.flush()
        except OSError:
            return
        time.sleep(0.25)


@pytest.fixture
def server(monkeypatch):
    instance = _Server(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv(
        "GOOGLE_GEMINI_BASE_URL", f"http://127.0.0.1:{instance.server_address[1]}"
    )
    yield instance
    instance.shutdown()
    instance.server_close()


def _answerer():
    settings = Settings(
        _env_file=None,
        gemini_api_key="k",
        answer_timeout_seconds=TIMEOUT_SECONDS,
    )
    return GeminiAnswerer(settings)


def _timed_call():
    started = time.monotonic()
    try:
        _answerer().answer("q", HITS)
    except AnswerError as e:
        return e, time.monotonic() - started
    pytest.fail("Expected the stalled call to fail")


def test_long_retry_after_stops_the_run_instead_of_sleeping(server):
    server.behavior = _rate_limited_with_long_retry_after

    error, elapsed = _timed_call()

    assert isinstance(error, AnswerQuotaError)
    assert elapsed < ALLOWED_SECONDS
    assert server.requests_seen == 1


def test_server_that_never_replies_hits_the_timeout(server):
    server.behavior = _never_replies

    error, elapsed = _timed_call()

    assert "Timeout" in str(error)
    assert elapsed < ALLOWED_SECONDS


def test_server_that_trickles_bytes_hits_the_timeout(server):
    server.behavior = _trickles_bytes

    error, elapsed = _timed_call()

    assert "Timeout" in str(error)
    assert elapsed < ALLOWED_SECONDS
