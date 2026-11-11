# 채점 자료

## a02

### 질문

파드 여러 대가 DB 한 대를 같이 쓰는데, DB가 멈춰도 파드가 한꺼번에 트래픽에서 빠지지 않게 하려면 뭘 바꿔야 하나요?

### 답변

모든 파드가 하나의 DB를 공유하는 환경에서는 `BALDUR_HEALTH_CHECK_READINESS_TIMEOUT_FAIL_DIRECTION` 환경 변수의 값을 `ready`로 설정해야 합니다. (기본값은 `not_ready`입니다.) 이 설정을 `ready`로 바꾸면 DB 응답 지연(stall) 발생 시 서비스 전체 파드가 한꺼번에 로테이션에서 제외(depool)되는 것을 막을 수 있습니다.

### 인용 문단

#### [3]

<passage>
제목: Health Check > How it works in Baldur

- **Degraded never depools.** Only `unhealthy` maps to `503` on the main endpoint (two rare
  internal-failure statuses, `error` and `unavailable`, also map to `503`). A degraded pod that can
  still serve correctly is deliberately kept in rotation — depooling healthy capacity because a
  background subsystem hiccupped would make an incident worse, not better.
- **The payload is a diagnosis, not a verdict.** Beyond the status, `health/` reports the default
  database's verdict, whether the self-healing automation is switched on, and a timestamp; with
  PRO self-monitoring active it adds each watched component's state. The pool endpoint
  additionally carries the error message when its check fails.
- **Readiness answers within a budget, whatever the database does.** A database that refuses
  connections fails fast and readiness reports it. A database that accepts the connection and
  then never answers is the dangerous case: without a bound, the probe itself hangs, your
  orchestrator's probe timeout expires, and the pod is depooled by default (with a shared
  database, every pod at once). Baldur probes all configured databases in parallel under one
  deadline and reports a stalled one as `timed_out`, so the verdict arrives on time and names
  the culprit. Whether a stall depools the pod is yours to choose, via
  `BALDUR_HEALTH_CHECK_READINESS_TIMEOUT_FAIL_DIRECTION`. Only readiness has this bound: a fresh
  `health/` computation and every `health/pool/` call wait as long as the database driver does,
  so point orchestrator probes at `health/ready/` and `health/live/`.
- **Responses are cached on purpose.** The full verdict comes from a precomputed cache so that
  aggressive probe polling stays cheap. Readiness is cached too, briefly and for the same
  reason: probe cadence times pod count should not mean constant query load against a database
  that may already be struggling. Append `?nocache=true` to force a fresh computation on
</passage>

#### [4]

<passage>
제목: Public Environment Variables (Operator-Tunable Allowlist) > Health check

## Health check

Readiness probes every configured database under a bounded budget. A database
that *refuses* connections always fails readiness. This variable decides the
other case: a database that accepts the connection but never answers, and so
exceeds the probe budget.

`not_ready` (the default) depools the pod, fast and honestly — the same outcome
a hung probe reaches today through the orchestrator's own probe timeout, but
decided by Baldur and visible in the response body. Choose `ready` when every
pod shares one database: there, depooling on a database stall takes the whole
service out of rotation at once, and staying in rotation degraded is the better
failure mode. Either way the affected alias is reported as `timed_out` in the
readiness body, so the stall is never silent.

```bash
BALDUR_HEALTH_CHECK_READINESS_TIMEOUT_FAIL_DIRECTION=not_ready
```
</passage>

## a03

### 질문

While a worker is shutting down gracefully, what does the readiness endpoint return so Kubernetes stops sending it new connections?

### 답변

During a graceful shutdown (DRAINING phase), the readiness endpoints flip to an HTTP 503 response containing a JSON payload with "status": "draining". This causes Kubernetes' readinessProbe to mark the pod as NotReady, preventing new connections from arriving.

### 인용 문단

#### [1]

<passage>
제목: Gunicorn Graceful Shutdown Runbook > Retry-After Semantics > Health-bridge readiness

### Health-bridge readiness

`/api/baldur/health/l3/` and `/api/baldur/health/bridge/` (the DB-independent readiness endpoints) flip to 503 during DRAINING with a `status: draining` payload:

```json
{
  "status": "draining",
  "shutdown": {
    "phase": "draining",
    "retry_after_seconds": 27,
    "in_flight_count": 3
  },
  ...
}
```

This makes k8s `readinessProbe` flip the pod's endpoint slice to NotReady, which deregisters it from the Service's load balancer — new connections stop arriving immediately, and only the in-flight requests counted in `in_flight_count` need to drain before the worker exits cleanly.

---
</passage>

#### [3]

<passage>
제목: Graceful Shutdown > How it works in Baldur

| What you observe | When it happens |
|------------------|-----------------|
| New requests get `503` with a `Retry-After` header and `Connection: close` | The drain has begun, on Django with Baldur's middleware installed (see *On Django* below). `Retry-After` carries the remaining drain time, and `Connection: close` makes load-balancer pools stop reusing the worker's connections |
| Baldur's liveness and ping endpoints keep answering `200` | The same middleware exempts them: draining is a normal lifecycle phase, not a failure, and keeping liveness green prevents Kubernetes from hard-killing the pod mid-drain. Other paths, readiness included, get the `503`, so new traffic routes elsewhere. Point the liveness probe at Baldur's endpoint, not one of your own |
| The process stays up while in-flight requests finish | The coordinator re-checks every half-second until every tracked request and every participating subsystem reports done. Standalone, it then re-delivers `SIGTERM` the conventional way and the process exits. Where a handler already owned the signal (uvicorn's, your own, or Python's `Ctrl+C` handler, which raises `KeyboardInterrupt` at once), Baldur starts the drain and passes the signal on, and that handler decides when the process exits; under gunicorn, the worker lifecycle does |
| `baldur_shutdown_phase` metric moves `0 → 1 → 3` | A clean drain: running, draining, terminated — nothing was lost |
| `baldur_shutdown_phase` reaches `2` | The drain window (30 seconds by default) expired with work still pending, and the drain was forced. `baldur_shutdown_aborted_requests_total` grows by the number of tracked requests still in flight at the deadline; a drain held open only by a subsystem adds nothing to it |
| `baldur_shutdown_drain_duration_seconds` histogram records the drain | Every shutdown reports how long it actually took — worth an alert if it creeps toward the window limit |
</passage>

## a04

### 질문

Which frameworks have a getting started guide?

### 답변

The frameworks that have a getting started guide are Django, FastAPI, Flask, and Celery.

### 인용 문단

#### [1]

<passage>
제목: Getting Started > Pick your framework

## Pick your framework

- [Django](django.md)
- [FastAPI](fastapi.md)
- [Flask](flask.md)
- [Celery](celery.md) — background tasks (workers + scheduled jobs)

Each quickstart assumes you have used the target framework at least once, and
ends with a short "Going to production" appendix: the shared cache backend
(Redis) that the zero-config path leaves out and a multi-worker deployment needs,
the production declaration that makes Baldur enforce it, and what PRO production
needs on top.
</passage>

## a06

### 질문

생존 신호 유예 시간을 정할 때 데몬 워커 반복 시간 지표를 쓰면 왜 안 되나요?

### 답변

생존 신호 유예 시간을 산정할 때 `baldur_daemon_worker_iteration_duration_seconds` 지표를 대신 사용하면 안 되는 이유는 해당 지표의 기본 히스토그램 버킷이 10초의 유한 한계(finite bound)까지만 설정되어 있기 때문입니다. 이로 인해 유예 시간 설정 시 파악해야 하는 수십 초에서 몇 분 동안 지연(stalled)된 모든 패스가 `+Inf`로 축소되어 지표에서 사라지게 됩니다.

### 인용 문단

#### [1]

<passage>
제목: Meta-Watchdog Escalation Response Runbook > Outbound liveness beacon (dead-man's switch) > Sizing the grace period — measure it, do not derive it

1. **It needs Prometheus, so this is a sizing-time procedure** — run it before (or independently of) turning the beacon on. It is never the incident-time signal; the beacon exists precisely because that stack can be absent when you need it.
2. **It under-reports by up to one scrape interval.** The 3× multiplier and the 10-minute floor absorb that.
3. **Do not substitute `baldur_daemon_worker_iteration_duration_seconds`.** It measures the same pass, but its default histogram buckets top out at a 10 s finite bound, so every stalled pass this rule cares about — tens of seconds to minutes — collapses into `+Inf` and disappears.

Once the beacon is live, your provider's own ping history is a valid cross-check.
</passage>

## a11

### 질문

장고 예제를 실행하면 마이그레이션이 안 됐다는 안내가 뜨는데, migrate를 해도 왜 안 사라지나요?

### 답변

해당 예제의 설정 파일이 데이터베이스를 인메모리(`:memory:`)로 유지하고 있어, 프로세스를 시작할 때마다 매번 비어 있는 데이터베이스로 새로 시작하기 때문입니다. 따라서 `manage.py migrate`를 실행해도 다음 프로세스 실행 시 다시 새로운 인메모리 데이터베이스가 생성되므로 안내 메시지가 사라지지 않습니다. 이 안내 메시지를 없애려면 `DATABASES` 설정을 파일이나 실제 서버를 가리키도록 변경한 후 마이그레이션을 진행해야 합니다.

### 인용 문단

#### [1]

<passage>
제목: Django Quickstart > 4. Run it

## 4. Run it

Start the dev server and call the route:

```bash
python manage.py runserver
curl http://127.0.0.1:8000/demo/
# {"status": "ok", "service": "demo"}
```

That's it. The response just travelled through a circuit breaker.

!!! note "The unapplied-migrations notice at startup is expected here"

    Baldur ships Django models for its own admin API, and this settings file
    keeps the database at `:memory:`, so every start begins with an empty one
    and Django says so. The demo route never touches the database, so the
    notice is harmless — and `manage.py migrate` will not silence it, because
    the next process gets a fresh in-memory database again. Point `DATABASES`
    at a file or a real server and the notice goes away once you migrate.
</passage>

## a18

### 질문

If my code puts items on the PRO in-memory replay queue, will Baldur take them off and process them for me?

### 답변

No. Nothing in Baldur dequeues from the PRO in-memory replay queue; the code that enqueues entries must also dequeue and process them.

### 인용 문단

#### [1]

<passage>
제목: baldur_pro.services.replay — Replay Queue

# baldur_pro.services.replay — Replay Queue

`ReplayQueueService`: an in-memory queue with token-bucket rate limiting and
`BackpressureStatus` / `RateLimitStatus` signals. Nothing in Baldur dequeues
from it; code that enqueues also dequeues and processes the entries.

Replaying stored failures is not PRO-only. The OSS `ReplayService`
([Service access](../services/access.md)) handles single-entry replay, batch
replay by failure type, and the automatic sweep that runs when a circuit
breaker recovers, and that replay does not go through this queue. See
[DLQ + Replay](../../concepts/foundations/dlq-replay.md) for the tier split.

!!! info "🔒 PRO Feature — requires a baldur-pro license"
    These symbols ship in the `baldur-pro` distribution. PRO modules import
    normally — there is no `ImportError`. PRO features activate only when
    `baldur.init()` runs with a valid `BALDUR_LICENSE_KEY`; without it the system
    runs with OSS defaults and `register_pro_services()` logs
    `entitlement.pro_registration_skipped`.

::: baldur_pro.services.replay
</passage>


## q01

### 질문

Baldur는 정확히 어떤 상황에서 쓰는 라이브러리인가요?

### 답변

Baldur는 결제 서비스, 데이터베이스, 외부 API(OpenAI, 이메일 서비스 등)와 같은 외부 의존 요소에서 타임아웃, 응답 지연, 에러 등의 장애가 발생하는 상황에 사용되는 Python 애플리케이션용 자가치유(self-healing) 신뢰성 라이브러리입니다. 데코레이터 하나로 서킷 브레이커, 재시도, 폴백, 데드 레터 큐 등의 복구 파이프라인을 구축하여 외부 장애가 사용자 에러나 온콜 호출로 이어지지 않고 자동으로 차단·재시도·복구되도록 보장합니다.

### 인용 문단

#### [4]

<passage>
제목: Baldur

# Baldur

[![CI](https://github.com/baldurhq/baldur/actions/workflows/ci-oss-mirror.yml/badge.svg)](https://github.com/baldurhq/baldur/actions/workflows/ci-oss-mirror.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![PyPI](https://img.shields.io/pypi/v/baldur-framework.svg)](https://pypi.org/project/baldur-framework/)
[![Docs](https://img.shields.io/badge/docs-baldur.sh-1f6feb.svg)](https://baldur.sh)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/13522/badge)](https://www.bestpractices.dev/projects/13522)

[English](https://github.com/baldurhq/baldur/blob/main/README.md) | **한국어**

> **초기 사용자 피드백을 받고 있습니다.** 실제 서비스에 붙여 보다가 설치·문서·예상과 다른 동작 등 막히는 곳이 있으면 [Discussions](https://github.com/baldurhq/baldur/discussions)나 [이슈](https://github.com/baldurhq/baldur/issues/new/choose)로 알려 주세요.

**여러분이 기대고 있는 외부 API가 한 시간 동안 죽으면, 여러분의 앱에는 무슨 일이 생기나요?**

요청은 타임아웃까지 매달려 있고, 워커는 전부 차 버리고, 그 한 시간 동안 실패한
작업은 사라집니다. OpenAI든, 결제 제공자든, 이메일 서비스든 — Baldur는 이 세
가지를 데코레이터 하나로 해결합니다. 온콜 담당자가 따로 없는 Python 서비스를
위해서요.

```python
import baldur
from openai import OpenAI

llm = baldur.llm.wrap(OpenAI(), timeout=60.0)


@baldur.protected("summarize", replay=True)
def summarize(doc_id: str) -> str:
    response = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": load_document(doc_id)}],
    )
    return response.choices[0].message.content
```

Redis도, Docker도, 설정도 없이 시작합니다. 저 두 줄은 멀티 프로세스로 가기
전까지 인메모리로 동작합니다. LLM을 부르지 않나요? 데코레이터만으로도 [어떤
의존성이든](#같은-데코레이터-어떤-의존성이든) 똑같이 됩니다.

트래픽이 흐르는 중에 제공자가 호출 한도를 걸거나, 죽거나, 그냥 느려지기만 해도:
</passage>

#### [5]

<passage>
제목: What is Baldur?

# What is Baldur?

> A self-healing reliability layer for Python applications — circuit breaker,
> retry, fallback, and dead-letter queue behind one decorator.

Every application depends on things that fail: a payment gateway times out, a
database gets slow, a third-party API starts returning errors. Baldur wraps the
calls you care about in a composed resilience pipeline, so those failures are
contained, retried, and recovered from automatically — instead of becoming
customer-visible errors and after-hours pages.

```python
import baldur


@baldur.protected("charge-customer")
def charge(order_id: str) -> dict:
    return payment_gateway.charge(order_id)
```

That decorator is the whole integration. With zero configuration it runs on an
in-memory backend — no Redis, no Docker, no environment variables.
</passage>

## q03

### 질문

FastAPI로 만든 서비스에도 Baldur를 적용할 수 있나요?

### 답변

네, FastAPI로 만든 서비스에도 Baldur를 적용할 수 있습니다. Baldur는 FastAPI 0.100 이상 버전을 지원하며, `baldur-framework[fastapi]` 패키지를 통해 FastAPI 경로를 보호할 수 있습니다.

### 인용 문단

#### [1]

<passage>
제목: FastAPI Quickstart

# FastAPI Quickstart

Protect a FastAPI route with Baldur. No Redis, no Docker, no
environment variables. The in-memory fallback covers the whole first run.

> Supports Python 3.11–3.13. Assumes you have built a FastAPI app before.
</passage>

#### [3]

<passage>
제목: Baldur > 호환성

## 호환성

| 구성 요소 | 최소 버전 | CI 테스트 대상 |
|-----------|-----------|----------------|
| Python | 3.11 | 3.11 · 3.12 · 3.13 |
| Django | 4.2 | 4.2 LTS · 5.2 LTS · 6.0 |
| FastAPI | 0.100 | 최소 버전 이상 최신 (스모크) |
| Flask | 2.3 | 최소 버전 이상 최신 (스모크) |
| Celery | 5.3 | 5.4 |
| Redis 서버 | — | 7.x |

전체 매트릭스와 Python × Django 테스트 그리드, 버전 지원 정책은
[호환성](https://baldur.sh/compatibility/)을 참고하세요.
</passage>

#### [4]

<passage>
제목: Baldur > 문서

## 문서

전체 문서는 **<https://baldur.sh>** 에 있습니다.

- [What is Baldur?](https://baldur.sh/what-is-baldur/) — 어떤 문제를 어떻게 푸는지
- 시작하기: [Django](https://baldur.sh/getting-started/django/) ·
  [FastAPI](https://baldur.sh/getting-started/fastapi/) ·
  [Flask](https://baldur.sh/getting-started/flask/) ·
  [Celery](https://baldur.sh/getting-started/celery/)
- [개념 가이드](https://baldur.sh) — 기능당 한 페이지, 이 README 전반에서 링크
- [API 레퍼런스](https://baldur.sh/reference/)
- [문제 해결](https://baldur.sh/troubleshooting/)
- [호환성](https://baldur.sh/compatibility/)
</passage>

#### [5]

<passage>
제목: Compatibility > Web frameworks

## Web frameworks

Baldur's core is framework-agnostic; the framework adapters are optional extras.

| Framework | Extra | Minimum | Tested in CI |
|-----------|-------|---------|--------------|
| Django | `baldur-framework[django]` | 4.2 | 4.2 LTS · 5.2 LTS · 6.0 · 6.1 |
| FastAPI | `baldur-framework[fastapi]` | 0.100 | latest ≥ floor (smoke) |
| Flask | `baldur-framework[flask]` | 2.3 | latest ≥ floor (smoke) |

Django is tested against the 4.2 and 5.2 LTS lines plus the 6.0 and 6.1
feature releases. FastAPI and Flask run a quickstart smoke test (install the extra, start
the app, hit a protected endpoint) against the latest release satisfying the
floor; that lane runs whenever the package or a quickstart app changes rather
than on every commit.

`[django]` installs Django only; mounting Baldur's REST API
(`baldur.api.django.urls`) also needs `baldur-framework[django-api]`.
</passage>

## q05

### 질문

재시도 횟수나 간격은 개발자가 직접 설정할 수 있나요?

### 답변

네, 환경 변수를 통해 직접 설정할 수 있습니다. `BALDUR_RETRY_MAX_ATTEMPTS`로 최대 시도 횟수를 설정할 수 있으며, `BALDUR_RETRY_BASE_DELAY`로 첫 대기 시간을, `BALDUR_RETRY_BACKOFF_STRATEGY`로 백오프 곡선 전략(exponential, linear, constant, decorrelated_jitter)을 설정할 수 있습니다.

### 인용 문단

#### [2]

<passage>
제목: Retry with exponential backoff in Python > Configuration

## Configuration

The most common knobs an operator sets. The full list lives in the API reference.

| Env Var | Default | What it controls |
|---------|---------|------------------|
| `BALDUR_RETRY_MAX_ATTEMPTS` | `3` | The maximum number of attempts, the first one included, before retry gives up on the operation |
| `BALDUR_RETRY_BASE_DELAY` | `1.0` | The first backoff wait; later waits follow the chosen strategy |
| `BALDUR_RETRY_BACKOFF_STRATEGY` | `exponential` | Which backoff curve the waits follow: `exponential`, `linear`, `constant` or `decorrelated_jitter` |
| `BALDUR_IDEMPOTENCY_ENABLED` | `true` | Master switch for duplicate-call blocking. Covers both `@idempotent` and `idempotency_key=` on the facade; set it to `false` and neither one checks or records a key |
</passage>

## q08

### 질문

기존 HTTP 클라이언트 코드를 많이 수정하지 않고 도입할 수 있을까요?

### 답변

네, 기존 코드를 많이 수정하지 않고 도입할 수 있습니다. Baldur는 결제 게이트웨이, 데이터베이스, 이메일 제공자 등 어떤 의존성이든 데코레이터 하나로 감싸는 방식을 사용하여 호출부를 전혀 바꾸지 않고 적용할 수 있습니다. 또한 별도의 설정이나 Redis, Docker 없이도 인메모리로 바로 시작할 수 있습니다.

### 인용 문단

#### [2]

<passage>
제목: Baldur > 같은 데코레이터, 어떤 의존성이든

## 같은 데코레이터, 어떤 의존성이든

결제 게이트웨이, 데이터베이스, 이메일 제공자 — 호출부는 전혀 바뀌지 않습니다.

```python
@baldur.protected("charge-customer", dlq=True)
def charge(order_id: str, amount_cents: int) -> dict:
    # 기본적으로 서킷 브레이커로 감싸집니다. dlq=True는 끝내 실패한 호출을
    # 인자와 함께 보관해서, 게이트웨이가 돌아오면 다시 실행할 수 있게 합니다.
    return payment_gateway.charge(order_id, amount_cents)
```

게이트웨이가 죽으면 브레이커가 열리고, 서비스는 타임아웃을 쌓아 올리는 대신 즉시
응답합니다. 나가는 길에 실패한 결제는 데드레터 큐에서 기다리며 내장 콘솔에
표시됩니다. 결제를 두 번 실행하면 두 번 청구될 수 있으므로, 보관된 결제는 사용자가
정한 방식으로만 돌아옵니다. `charge-customer`에 어떤 결제가 다시 실행해도
안전한지 판단하는 재실행 핸들러를 등록하면, 콘솔에서 클릭 한 번으로, 또는 허용한
실패 유형에 한해 브레이커가 닫힐 때 자동으로 재실행됩니다 — 아래 데모가 바로
그렇게 합니다. 두 번 실행해도 괜찮은 작업이라면 `replay=True`만으로 핸들러 없이
됩니다. (재실행은 나가는 길에 실패한 작업을 위한 것이지, 비즈니스 상의 거절이나
고객이 이미 떠나버린 결제를 위한 것이 아닙니다 —
[그 경계가 어디인지](https://baldur.sh/concepts/foundations/dlq-replay/).)

![터미널 데모: 트래픽이 흐르는 중에 결제 게이트웨이가 응답 불능이 되고 결제 1,000건이 들어옵니다 — 5건은 게이트웨이까지 가서 실패하고, 브레이커가 열려 나머지 995건을 그 자리에서 거절하며, 1,000건이 전부 포착되어 복구 시점에 데모의 재실행 핸들러를 통해 10번에 나눠 1,000건이 전부 재실행됩니다. 유실 0건.](https://raw.githubusercontent.com/baldurhq/baldur/main/.github/assets/demo-payment-outage.gif)

*결제 데모입니다. 트래픽이 흐르는 중에 게이트웨이가 응답 불능이 되고 결제
1,000건이 들어옵니다. 5건은 게이트웨이까지 가서 실패하고, 열린 브레이커가 나머지
995건을 게이트웨이를 부르지 않고 거절하며, 1,000건이 전부 인자와 함께 포착되어
데모가 등록한 재실행 핸들러를 통해 복구 시점에 전부 재실행됩니다. 유실 0건. 실제 실행을 실제 시간 그대로 녹화한
화면이고, 브레이커 상태와 DLQ 집계는 프레임워크에서 실시간으로 읽어온 값입니다.
데코레이터 자체는 `pip install baldur-framework`가 전부입니다. 데모는 프로세스
안의 대역 워커를 위해 `celery` extra를 추가할 뿐, 여전히 프로세스 하나에 Redis도
브로커도 없습니다. 직접 재현해 보세요:*

```bash
pip install "baldur-framework[celery]"
python -m baldur.scripts.demo_self_healing --outage-charges 1000
```

기본값 이상이 필요하다면 파이프라인을 선언적으로 조합하면 됩니다.

```python
@baldur.protected(
    "summarize",
    timeout=30.0,                            # 호출자가 기다리는 시간의 단일 상한
    fallback=lambda: last_good_summary(),    # OPEN 상태일 때의 우아한 응답
    idempotency_key="doc_id",                # 재전달된 작업도 한 번만 처리
)
def summarize(doc_id: str) -> str:
    return llm_api.summarize(doc_id)
```
</passage>

#### [3]

<passage>
제목: Baldur

# Baldur

[![CI](https://github.com/baldurhq/baldur/actions/workflows/ci-oss-mirror.yml/badge.svg)](https://github.com/baldurhq/baldur/actions/workflows/ci-oss-mirror.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![PyPI](https://img.shields.io/pypi/v/baldur-framework.svg)](https://pypi.org/project/baldur-framework/)
[![Docs](https://img.shields.io/badge/docs-baldur.sh-1f6feb.svg)](https://baldur.sh)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/13522/badge)](https://www.bestpractices.dev/projects/13522)

[English](https://github.com/baldurhq/baldur/blob/main/README.md) | **한국어**

> **초기 사용자 피드백을 받고 있습니다.** 실제 서비스에 붙여 보다가 설치·문서·예상과 다른 동작 등 막히는 곳이 있으면 [Discussions](https://github.com/baldurhq/baldur/discussions)나 [이슈](https://github.com/baldurhq/baldur/issues/new/choose)로 알려 주세요.

**여러분이 기대고 있는 외부 API가 한 시간 동안 죽으면, 여러분의 앱에는 무슨 일이 생기나요?**

요청은 타임아웃까지 매달려 있고, 워커는 전부 차 버리고, 그 한 시간 동안 실패한
작업은 사라집니다. OpenAI든, 결제 제공자든, 이메일 서비스든 — Baldur는 이 세
가지를 데코레이터 하나로 해결합니다. 온콜 담당자가 따로 없는 Python 서비스를
위해서요.

```python
import baldur
from openai import OpenAI

llm = baldur.llm.wrap(OpenAI(), timeout=60.0)


@baldur.protected("summarize", replay=True)
def summarize(doc_id: str) -> str:
    response = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": load_document(doc_id)}],
    )
    return response.choices[0].message.content
```

Redis도, Docker도, 설정도 없이 시작합니다. 저 두 줄은 멀티 프로세스로 가기
전까지 인메모리로 동작합니다. LLM을 부르지 않나요? 데코레이터만으로도 [어떤
의존성이든](#같은-데코레이터-어떤-의존성이든) 똑같이 됩니다.

트래픽이 흐르는 중에 제공자가 호출 한도를 걸거나, 죽거나, 그냥 느려지기만 해도:
</passage>

## q09

### 질문

실패한 요청을 나중에 다시 처리하는 기능도 있나요?

### 답변

네, 실패한 작업을 보관한 뒤 나중에 다시 실행할 수 있는 'DLQ + Replay(데드 레터 큐 및 재실행)' 기능이 있습니다. 종속 시스템이나 서비스 장애로 실패한 작업은 인자와 함께 보관되며, 콘솔에서 수동으로 재실행하거나 제공자가 복구되었을 때 셀러리(Celery) 워커를 통해 자동으로 다시 실행할 수 있습니다.

### 인용 문단

#### [2]

<passage>
제목: DLQ + Replay

# DLQ + Replay

> Catches the operations that fail because a dependency was down, at the call sites that opt in, with the context needed to run them again, so nothing is silently lost — and once the dependency recovers, the backlog replays.
</passage>

#### [3]

<passage>
제목: Baldur

- **워커들이 함께 물러섭니다.** 429나 "과부하" 응답이 오면 모든 워커가 공유하는
  대기가 하나 걸리고, 그 길이는 제공자가 요청한 시간 이상입니다. 첫 워커만 거절당하고
  나머지는 기다립니다 — 워커마다 따로 한도를 부딪쳐 알아내지 않습니다. 제공자가
  거절한 요청(400, 422)은 재시도하지 않습니다.
- **앱은 계속 응답합니다.** 매달린 요청은 60초 상한에서 실패로 바뀌고, 서킷
  브레이커가 열리며, 호출은 즉시 실패합니다 — 느려진 제공자 하나가 워커 전체를
  끌고 내려가지 않습니다. wrap에 `fallbacks=[...]`를 주면 호출이 다음 엔드포인트로
  넘어갑니다(OpenAI SDK, Anthropic SDK, google-genai).
- **실패한 작업은 사라지지 않고 보관됩니다.** 끝내 실패한 호출은 전부 인자와
  함께 포착되어 내장 콘솔(`http://127.0.0.1:9090/`)에 목록으로 남습니다.
  프레임워크 없이 쓰는 파이썬 프로세스에서는 시작할 때 `baldur.init()`을 한 번
  호출해야 콘솔이 뜹니다. Django·FastAPI·Flask 연동은 이 호출을 알아서 해 줍니다.
- **그리고 돌아옵니다.** `replay=True`면 Baldur가 보관된 작업을 저장된 인자로
  다시 실행합니다. 콘솔에서 클릭 한 번으로, 또는 제공자가 복구되어 그 작업의
  브레이커가 닫히는 순간 자동으로 — 재실행은 Celery 워커가 돌립니다.

실제 `openai` SDK와 로컬 가짜 제공자로 직접 보세요 — 호출 한도, 이어서 장애,
그리고 보관된 작업 전부의 재실행:

![터미널 데모: 워커 8개가 LLM 제공자의 429(retry-after 5)를 만납니다 — SDK 자체 재시도로는 대기 시간 안에 요청 8건이 제공자에 도착하고, baldur.llm.wrap을 거치면 1건입니다. 이어서 제공자가 503을 돌려주자 작업 12건이 인자와 함께 보관되고, 제공자가 복구되어 브레이커가 닫히면 12건 모두 자동으로 다시 실행됩니다. 유실 0건.](https://raw.githubusercontent.com/baldurhq/baldur/main/.github/assets/demo-llm-outage.gif)

*실제 실행을 실제 시간 그대로 녹화했습니다. 직접 재현해 보세요:*

```bash
pip install "baldur-framework[celery]" openai
python -m baldur.scripts.demo_llm_outage
```

**Baldur가 고장 나면 내 호출도 실패하나요?** 아닙니다. 실패한 작업을 보관하다가
오류가 나는 식으로 Baldur 자체 기록 작업이 실패하면 로그로만 남고, 호출은 원래
결과를 돌려주거나 원래 오류를 던집니다. Redis에 연결할 수 없으면 호출은 각
프로세스의 자체 상태로 계속 돌고, Redis가 돌아올 때까지 워커들이 429 대기와
브레이커를 공유하지 못할 뿐입니다. 예외가 하나 있고 의도된 동작입니다.
`idempotency_key=`를 단 호출은 두 번 실행을 막아 주는 공유 기록 없이 실행하는
대신, 몇 초 기다린 뒤 `IdempotencyUnavailableError`로 거절됩니다.

**재시도나 재실행 때문에 요금이 두 번 나갈 수 있나요?** 네, 그러니 대비해야
합니다. 재실행은 작업 전체를 처음부터 다시 돌립니다. 작업 안에서 이미 성공했던
모델 호출도 다시 나가고 다시 과금됩니다. 시간 초과로 끝난 요청도 제공자 쪽에서는
끝까지 처리됐을 수 있어서, 재시도하면 두 번 과금될 수 있습니다. `replay=True`는
두 번 돌아도 괜찮은 작업에만 쓰거나, 단계마다 결과를 직접 저장해서 이미 끝난
단계는 건너뛰게 하세요.

Django, FastAPI, Flask, Celery 어댑터가 들어 있습니다.
</passage>

#### [4]

<passage>
제목: DLQ + Replay > What is it?

## What is it?

When an operation in your app fails because something it relies on is unavailable (a database
timeout, a payment provider returning 503, a webhook endpoint that won't answer), that unit of work
usually just disappears. The request errors out, maybe a line lands in a log file, and the actual
work (the payment, the job, the message) is lost.

A **dead letter queue** (DLQ) is the standard name for a holding area where failed work is parked
instead of thrown away. Think of the "undeliverable mail" bin at a post office: a letter that
couldn't be delivered isn't shredded; it's set aside so it can be looked at and sent again later.
**Replay** is the act of taking that parked work and running it again. In Baldur these two halves
are a single feature: **DLQ + Replay**.
</passage>

## q12

### 질문

여러 서버 인스턴스에서 동시에 실행해도 문제가 없나요?

### 답변

여러 Pod/호스트(여러 서버 인스턴스) 환경에서 일관성(coherence)과 단일 실행을 보장하려면 추가 설정이 필요합니다. Redis 기반 설정을 적용하거나 Celery beat을 대체 사용하는 방법이 있습니다.

1. **Redis 설정 적용 (baldur-pro 필요)**:
   - `BALDUR_SYSTEM_CONTROL_BACKEND=redis` (공유 상태 백엔드)
   - `BALDUR_EVENT_BUS_BACKEND=redis` (포드 간 CONFIG_UPDATED 전달)
   - `BALDUR_LEADER_ELECTION_ENABLED=1` 및 `BALDUR_LEADER_ELECTION_BACKEND=redis` (클러스터 단위의 단일 스케줄러 보장)
   - `BALDUR_REDIS_URL=redis://your-redis:6379/0` 지정

2. **대안 방법**:
   - 내장 스케줄러 대신 Celery beat을 실행하고 `BALDUR_SCHEDULER_AUTOSTART=0`으로 설정합니다.

### 인용 문단

#### [4]

<passage>
제목: Multi-Worker Coherence Runbook > TL;DR

## TL;DR

- **Single host (default)** — works out-of-box, no external infra. One process per host elects itself via a local file lock and runs the scheduled jobs (including applying DELAYED/GRACEFUL config changes, on installs with an active PRO entitlement). Nothing to configure.
- **Multiple pods/hosts** — set all three:

  ```bash
  export BALDUR_SYSTEM_CONTROL_BACKEND=redis     # shared state backend (the default once BALDUR_REDIS_URL is set)
  export BALDUR_EVENT_BUS_BACKEND=redis          # cross-pod CONFIG_UPDATED
  export BALDUR_LEADER_ELECTION_ENABLED=1        # one scheduler cluster-wide
  export BALDUR_LEADER_ELECTION_BACKEND=redis
  export BALDUR_REDIS_URL=redis://your-redis:6379/0
  ```

  (Requires `baldur-pro` for the Redis leader elector.) **Or** run Celery beat instead of the inline scheduler and set `BALDUR_SCHEDULER_AUTOSTART=0`.

---
</passage>

## q16

### 질문

What's the main difference between Baldur and a regular retry decorator?

### 답변

Regular retry libraries only provide retry with backoff, which repeatedly hits a failing dependency. In contrast, Baldur combines retry with additional resilience features into a single decorator, including a circuit breaker (which trips and fails fast when a dependency stays down), fallbacks, a dead-letter queue for capturing and replaying failed work, idempotency guards, health checks, observability, and time limits on caller waits.

### 인용 문단

#### [1]

<passage>
제목: FAQ > How is this different from tenacity / backoff / retry libraries?

## How is this different from tenacity / backoff / retry libraries?

Those libraries give you **one pattern** — retry with backoff. Baldur
composes retry with a circuit breaker, fallback, dead-letter queue,
idempotency guard, health checks, and observability into a single
decorator. The difference shows up when a dependency stays down:

- A retry library keeps hammering it.
- Baldur's circuit breaker trips and fails fast; a fallback serves a
  safe answer, or — when no fallback is set — the failed work is
  captured in a DLQ for later replay. After a cool-down the breaker
  lets a few trial calls through and closes again once they succeed.

If you already use tenacity and want to keep it, Baldur includes a
[tenacity bridge](concepts/foundations/composition.md) that routes
tenacity-managed calls through the rest of the pipeline.

---
</passage>

#### [2]

<passage>
제목: Baldur

**Already using your SDK's retries?** Around a decorated call, keep them.
Baldur doesn't replace retry — it adds what retry can't: a breaker so one
incident doesn't cost every request its retries, one wall-clock bound on what
the caller waits, a fallback, and the capture-and-replay no retry library gives
you. A `baldur.llm.wrap` client is the one exception: it turns the SDK's own
retries off on its copy of the client, because there one coordinated retry
replaces them.
</passage>

## q18

### 질문

How do I configure Baldur to handle HTTP 429 responses?

### 답변

Baldur recognizes 429 responses automatically without manual wiring when using protected call wrappers like `@baldur.protected`, `@baldur.aprotected`, `protect()`, `aprotect()`, or the `circuit_breaker` decorator.

Configuration and handling options include:
- **Status Code List**: The status codes treated as rate limits are configured via `BALDUR_MIDDLEWARE_RATE_LIMIT_CODES` (defaults to `429`).
- **Enabling Retries for Returned 429s**: Raised 429 exceptions are retried automatically under `@baldur.protected(..., retry=True)`. Returned 429 response objects install the shared cooldown but are handed back without being retried. To retry returned 429s, configure the client to raise an exception (e.g., `response.raise_for_status()`).
- **Standalone Rate Limit Awareness**: For functions outside of `@baldur.protected`, use `@coordinator.rate_limit_aware("service_name")` from `RateLimitCoordinator.get_instance()` to wait out active cooldowns and report 429 responses without retrying.

### 인용 문단

#### [1]

<passage>
제목: Circuit breaker for Python > How it works in Baldur > When a dependency answers 429

### When a dependency answers 429

Baldur watches for a 429 at the point where the protected call is made, on every framework and in
both call styles: `@baldur.protected` and `@baldur.aprotected`, `protect()` and `aprotect()`, and
the `circuit_breaker` decorator on a `def` or an `async def`. Nothing has to be wired by hand. Both
shapes a client can deliver a 429 in are recognised: an exception whose message or type names the
rate limit, and a returned response object whose integer status code is on the rate-limit list, so a
`requests` call that hands the 429 back without raising is caught as well.

Each sighting does three things. Seen by the breaker stage, it records a breaker failure, so the
consecutive count and the failure rate move as they would for any exception. It feeds the
**rate-limit cascade**, the trigger
for storms interleaved with successes: with the defaults, ten 429s inside a minute open the breaker
once they make up at least a tenth of the calls made in that minute and the minute holds at least
twenty of them, counted per worker process. And it starts the cooldown that Baldur's retry stage
waits out before its next attempt, sized from the provider's `Retry-After` when the answer carried
one, so a retry does not land while the dependency is still asking for room.
</passage>

#### [4]

<passage>
제목: Retry and backoff for LLM API rate limits in Python > What a returned 429 gets, and what it does not

## What a returned 429 gets, and what it does not

A 429 reaches Baldur in one of two shapes: an exception the client raises
(`RateLimitError` from the OpenAI SDK, `httpx.HTTPStatusError` after
`raise_for_status()`), or a response object the client hands back with
`status_code == 429` — `requests` or `httpx` without `raise_for_status()`.

Both shapes install the shared cooldown, except a raised 429 the table above
reads as an exhausted quota. Where a raised exception is classified as above, a
returned response is classified by its status code (`429` by default;
`BALDUR_MIDDLEWARE_RATE_LIMIT_CODES` is the list). Either way the cooldown is
written, and the rest of the fleet waits on it.

The difference is the retry. A raised 429 is a failure, so the retry ladder
runs and, unless it is an exhausted quota, the call is tried again once the
cooldown has passed. A returned 429
is a value, so the retry stage hands it back to your code as the result:
coordinated, but not retried. If you want the retry as well, make the client
raise:

```python
@baldur.protected("openai_chat", retry=True)
def ask(prompt: str) -> str:
    response = httpx.post(...)
    response.raise_for_status()   # now the 429 is a failure the ladder retries
    return response.json()
```

For a call you do not want under `@baldur.protected` at all, the coordinator
can be driven directly:

```python
from baldur.services.rate_limit_coordinator import RateLimitCoordinator

coordinator = RateLimitCoordinator.get_instance()

@coordinator.rate_limit_aware("openai_chat")
def ask(prompt: str):
    return httpx.post(...)          # inspected for a 429 on the way back
```

That decorator waits out an active cooldown before calling, reads the returned
response (or the raised error) for a 429, and reports it. It does not retry
either. If the remaining cooldown is longer than it is willing to sleep, it
raises instead of calling — a decorator cannot return "nothing," so refusing is
its only honest option.
</passage>

#### [5]

<passage>
제목: Circuit breaker for Python > How it works in Baldur > When a dependency answers 429

Two things deliberately do not count. A call Baldur itself declined to make, because a cooldown
outlasted what the retry stage could wait, is never a breaker failure and never enters the
rate: the dependency was not contacted. And an exception type you told the breaker to ignore
(`ignore_exceptions`) is ignored for the cascade as well, at every depth: a 429 the retry stage
classifies on its own, in either call style, is dropped by the same list. The list names exception
types, so a 429 a client hands back as a returned response is always counted.

Which statuses count is set by two lists, `BALDUR_MIDDLEWARE_CB_STATUS_CODES` (failures,
`500,502,503,504` by default) and `BALDUR_MIDDLEWARE_RATE_LIMIT_CODES` (`429` by default). The prefix
is historical: the same two lists govern this outbound path and the inbound middleware. They are not
exclusive either. List `503` in both and an upstream that sheds load with 503s records a failure and
feeds the cascade at once.
</passage>

## q26

### 질문

How can I monitor whether Baldur is actually recovering failed API calls in production?

### 답변

You can monitor self-healing and recovery actions using metrics that Baldur records automatically whenever a circuit breaker changes state, a retry runs out of attempts, a dead-letter item is created or resolved, or a replay finishes. You can point your existing monitoring system to scrape these metrics from the Prometheus text-exposition page (served by default at `/prometheus` on port 9090, or at `prometheus/` under REST URLs in Django) or read the JSON view of the control-API metrics.

### 인용 문단

#### [1]

<passage>
제목: Metrics > How it works in Baldur

## How it works in Baldur

**Metrics are recorded automatically.** When a circuit breaker changes state, a retry runs out of
attempts, a dead-letter item is created or resolved, or a replay finishes, Baldur updates the
matching metric for you. You write no recording code: the numbers appear because the self-healing
layer is doing its job.

**Your monitoring system scrapes them in the standard format.** Baldur exposes its metrics as a
Prometheus text-exposition page (served byte-exact for scrapers) plus a JSON view of the
control-API metrics. Baldur's built-in admin server serves the page at `/prometheus` whichever
framework you run, and without one. It listens on port 9090 and on localhost only by default, so a
Prometheus server on another host or pod cannot reach it until you open the admin server up; the
[admin server settings](../../reference/env-vars.md#admin-server) cover the port and the key that a
non-localhost bind requires. On Django, Baldur's REST URLs (the `[django-api]` extra) also serve the
page at `prometheus/` under wherever you include them. Flask and FastAPI apps get no route of their
own. Either way you point your existing Prometheus server at it, with nothing Baldur-specific to
learn on the scraper side.

**You can instrument your own functions too.** Two decorators cover the common cases:

| Decorator | What it records |
|-----------|-----------------|
| `@track_counter` | Counts calls that return; pass `on_failure=True` to count calls that raise as well, and add `on_success=False` to count only those |
| `@track_execution_time` | Records how long a function takes, as a histogram |

**The Cardinality Guard keeps traffic-driven labels bounded.** This is the part that makes the
metrics safe to leave on in production:
</passage>
