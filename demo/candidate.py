"""Simulated LLM-generated module: a small statistics toolkit.

Planted defects (ground truth for evaluation):
  - `median`          : wrong index for even-length input        [REAL BUG]
  - `variance`        : divides by n instead of n-1              [REAL BUG]
  - `normalize`       : no guard for zero range -> ZeroDivision  [REAL BUG]
  - `percentile`      : off-by-one in index computation          [REAL BUG]
  - `mean`, `stdev`, `summarize` are correct but depend on the above.
"""


def mean(xs):
    return sum(xs) / len(xs)


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n % 2 == 0:
        # BUG: should average s[n//2 - 1] and s[n//2]
        return s[n // 2]
    return s[n // 2]


def variance(xs):
    m = mean(xs)
    # BUG: sample variance should divide by (n - 1)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def stdev(xs):
    return variance(xs) ** 0.5


def normalize(xs):
    lo = min(xs)
    hi = max(xs)
    # BUG: no guard when hi == lo
    return [(x - lo) / (hi - lo) for x in xs]


def percentile(xs, p):
    s = sorted(xs)
    # BUG: off-by-one, index can equal len(s)
    k = int(len(s) * p / 100)
    return s[k]


def summarize(xs):
    return {
        "mean": mean(xs),
        "median": median(xs),
        "stdev": stdev(xs),
        "p90": percentile(xs, 90),
    }
