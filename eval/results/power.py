"""Power of the sealed third set for a retrieval change, computed exactly.

Uses only the recorded count of answerable questions in set 3 (18). It never reads
the sealed question file and makes no API call.

Model: the baseline hits H of the 18 questions. The change fixes each baseline
miss with probability FIX and breaks each baseline hit with probability BREAK,
independently. b = fixed questions, c = broken ones. Two decision rules are scored:
  - net: b - c >= 2, the rule pre-registered for the reranker
  - sign test: exact one-sided McNemar (sign) test on the b + c discordant
    questions, p <= 0.05
Run from the repository root: python eval/results/power.py
"""

from math import comb

QUESTIONS = 18
ALPHA = 0.05
NET_THRESHOLD = 2
BASELINE_HITS = [9, 11, 13]  # set 2 hit 60% would be about 11 of 18
FIX_RATES = [0.0, 0.2, 0.3, 0.5, 0.7]
BREAK_RATES = [0.0, 0.05, 0.1]


def binom_pmf(n: int, k: int, p: float) -> float:
    return comb(n, k) * p**k * (1 - p) ** (n - k)


def sign_test_p(b: int, c: int) -> float:
    """One-sided exact p-value that b or more of b + c go the improving way."""
    d = b + c
    if d == 0:
        return 1.0
    return sum(comb(d, k) for k in range(b, d + 1)) / 2**d


def pass_rates(
    hits: int, fix: float, brk: float, questions: int = QUESTIONS
) -> tuple[float, float]:
    misses = questions - hits
    net = sign = 0.0
    for b in range(misses + 1):
        pb = binom_pmf(misses, b, fix)
        for c in range(hits + 1):
            p = pb * binom_pmf(hits, c, brk)
            net += p * (b - c >= NET_THRESHOLD)
            sign += p * (sign_test_p(b, c) <= ALPHA)
    return net, sign


print("Smallest wins that pass the sign test (one-sided, p <= 0.05):")
for c in range(4):
    b = next(b for b in range(QUESTIONS + 1) if sign_test_p(b, c) <= ALPHA)
    print(f"  {c} broken -> at least {b} fixed (p = {sign_test_p(b, c):.3f})")

print()
print("hits  fix  break  E[fixed]  E[broken]  P(net>=2)  P(sign test)")
for hits in BASELINE_HITS:
    for fix in FIX_RATES:
        for brk in BREAK_RATES:
            net, sign = pass_rates(hits, fix, brk)
            print(
                f"{hits:4} {fix:4.2f} {brk:6.2f} {fix * (QUESTIONS - hits):9.1f}"
                f" {brk * hits:10.1f} {net:10.2f} {sign:13.2f}"
            )

print()
print("No net effect (expected fixed = expected broken), baseline 11 hits:")
for churn in (0.5, 1.0, 2.0):
    net, sign = pass_rates(11, churn / 7, churn / 11)
    print(f"  {churn} fixed and {churn} broken expected: P(net>=2) {net:.2f}, sign {sign:.2f}")

print()
print("Answerable questions needed for 80% sign-test power, baseline hit 60%:")
for fix, brk in [(0.3, 0.05), (0.5, 0.05), (0.7, 0.05)]:
    n = next(
        n
        for n in range(QUESTIONS, 400)
        if pass_rates(round(0.6 * n), fix, brk, questions=n)[1] >= 0.8
    )
    print(f"  fix {fix:.2f}, break {brk:.2f}: {n} questions")
