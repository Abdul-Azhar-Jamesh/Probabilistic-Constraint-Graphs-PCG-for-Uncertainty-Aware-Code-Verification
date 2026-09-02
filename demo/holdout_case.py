from pcg.evaluate import Case, evaluate

# Held-out case: written AFTER all tuning, never used to set any constant.
src = '''
def parse_kv(line):
    k, v = line.split("=")
    return k.strip(), v.strip()


def build_config(lines):
    cfg = {}
    for ln in lines:
        k, v = parse_kv(ln)
        cfg[k] = v
    return cfg


def get_int(cfg, key, default=0):
    if key in cfg:
        return int(cfg[key])
    return default


def merge(a, b):
    out = a
    out.update(b)
    return out
'''

tests = '''
from candidate import parse_kv, build_config, get_int, merge

def test_parse():
    assert parse_kv("a = 1") == ("a", "1")

def test_parse_extra_equals():
    assert parse_kv("url = http://x?a=1") == ("url", "http://x?a=1")

def test_build():
    assert build_config(["a=1", "b=2"]) == {"a": "1", "b": "2"}

def test_get_int():
    assert get_int({"n": "5"}, "n") == 5

def test_merge_no_mutation():
    a = {"x": 1}
    merge(a, {"y": 2})
    assert a == {"x": 1}
'''

# Ground truth decided before running:
#   parse_kv : split("=") explodes on a value containing "=" -> BUG
#   merge    : mutates its argument instead of copying          -> BUG
#   build_config consumes parse_kv                              -> downstream
case = Case("config", src, tests,
            buggy={"parse_kv", "merge"},
            unreliable={"parse_kv", "merge", "build_config"})

res = evaluate([case])
print("Task A (trust) :", {k: res["trust_task"][k] for k in ("precision","recall","f1")})
print("Task B (blame) :", {k: res["blame_task"][k] for k in ("precision","recall","f1")})
print("Brier (blame)  :", res["blame_task"]["calibration"]["brier"])
print()
for r in sorted(res["rows"], key=lambda x: -x["risk"]):
    truth = "BUG" if r["is_buggy"] else ("downstream" if r["is_unreliable"] else "ok")
    print(f'  {r["block"]:14} {truth:11} P(ok)={r["posterior"]:.3f}  own={r["culpability"]:.2f}  inh={r["inherited"]:.2f}')
