"""Hand-specified synthetic families, independent of PCG predictions/mutators.

Each fixed version is checked against explicit expected behavior. Family splits
are fixed before fitting. These examples measure mechanics, not production accuracy.
"""

from __future__ import annotations


def curated_dataset() -> list[dict]:
    # name, split, source, old expression, fixed expression, defect, affected, oracle
    families = [
        (
            "discount",
            "train",
            "def total(price, rate):\n    return price * (1 + rate)\n",
            "1 + rate",
            "1 - rate",
            "total",
            [],
            "assert c.total(100, .2) == 80",
        ),
        (
            "empty_average",
            "train",
            "def average(xs):\n    return sum(xs) / len(xs)\n",
            "return sum(xs) / len(xs)",
            "return sum(xs) / len(xs) if xs else 0",
            "average",
            [],
            "assert c.average([]) == 0\n    assert c.average([2, 4]) == 3",
        ),
        (
            "boundary",
            "train",
            "def eligible(age):\n    return age > 18\n",
            "age > 18",
            "age >= 18",
            "eligible",
            [],
            "assert c.eligible(18) is True\n    assert c.eligible(17) is False",
        ),
        (
            "rare_type",
            "train",
            "def choose(n: int) -> int:\n    if n * 7 + 1 == 6560:\n        return 'bad'\n    return n\n",
            "return 'bad'",
            "return n",
            "choose",
            [],
            "assert c.choose(937) == 937",
        ),
        (
            "normalize",
            "validation",
            "def normalize(name):\n    return name.strip()\ndef key(name):\n    return normalize(name)\n",
            "name.strip()",
            "name.strip().lower()",
            "normalize",
            ["key"],
            "assert c.key(' Alice ') == 'alice'",
        ),
        (
            "mutable_default",
            "validation",
            "def add(value, xs=[]):\n    xs.append(value)\n    return xs\n",
            "xs=[]):\n",
            "xs=None):\n    xs = [] if xs is None else xs\n",
            "add",
            [],
            "assert c.add(1) == [1]\n    assert c.add(2) == [2]",
        ),
        (
            "prefix",
            "validation",
            "def remove_prefix(value, prefix):\n    return value.replace(prefix, '')\n",
            "value.replace(prefix, '')",
            "value[len(prefix):] if value.startswith(prefix) else value",
            "remove_prefix",
            [],
            "assert c.remove_prefix('ababx', 'ab') == 'abx'\n    assert c.remove_prefix('xaby', 'ab') == 'xaby'",
        ),
        (
            "nested_contract",
            "validation",
            "def coupon(code: str, n: int) -> int:\n    if code == 'LIVE' and n == 321:\n        return 'wrong'\n    return n\n",
            "return 'wrong'",
            "return n",
            "coupon",
            [],
            "assert c.coupon('LIVE', 321) == 321",
        ),
        (
            "delayed_division",
            "test",
            "def denominator(n):\n    value = n - 7\n    return value\ndef ratio(n):\n    return 100 / denominator(n)\n",
            "n - 7",
            "n + 7",
            "denominator",
            ["ratio"],
            "assert c.ratio(7) == 100 / 14",
        ),
        (
            "previous_state",
            "test",
            "class Counter:\n    def __init__(self):\n        self.value = 0\n    def add(self, n):\n        self.value = n\n    def read(self):\n        return self.value\n",
            "self.value = n",
            "self.value += n",
            "Counter.add",
            ["Counter.read"],
            "obj = c.Counter()\n    obj.add(2)\n    obj.add(3)\n    assert obj.read() == 5",
        ),
        (
            "list_alias",
            "test",
            "def ordered(xs):\n    xs.sort()\n    return xs\n",
            "xs.sort()\n    return xs",
            "return sorted(xs)",
            "ordered",
            [],
            "values = [3, 1, 2]\n    assert c.ordered(values) == [1, 2, 3]\n    assert values == [3, 1, 2]",
        ),
        (
            "helper_type",
            "test",
            "def helper(n: int) -> int:\n    if n * 3 + 2 == 2141:\n        return 'bad'\n    return n\ndef wrapper(n: int) -> int:\n    return helper(n)\n",
            "return 'bad'",
            "return n",
            "helper",
            ["wrapper"],
            "assert c.wrapper(713) == 713",
        ),
    ]
    rows = []
    for name, split, source, old, fixed, root, impacted, oracle in families:
        assert source.count(old) == 1
        tests = "import candidate as c\ndef test_behavior():\n    " + oracle + "\n"
        for version in ("buggy", "fixed"):
            rows.append(
                {
                    "name": f"{name}:{version}",
                    "repository": name,
                    "split": split,
                    "source": source
                    if version == "buggy"
                    else source.replace(old, fixed),
                    "tests": tests,
                    "buggy": [root] if version == "buggy" else [],
                    "unreliable": [root, *impacted] if version == "buggy" else [],
                    "provenance": "Hand-specified synthetic defect and independent behavior assertion; not a published real bug.",
                }
            )
    return rows
