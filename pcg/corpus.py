"""Correct reference programs used as mutation seeds.

Each entry is (source, tests) where the source is *correct* and passes every
test. Mutation operators are applied to these to synthesise labelled defects.

The programs are deliberately multi-block with real inter-block dependencies,
because a corpus of independent single functions would never exercise the
constraint-propagation half of the framework -- inherited doubt only appears
when one block consumes another.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
_STATS = (
    '''
def mean(xs):
    if not xs:
        return 0.0
    return sum(xs) / len(xs)


def variance(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def stddev(xs):
    return variance(xs) ** 0.5


def zscores(xs):
    s = stddev(xs)
    if s == 0:
        return [0.0 for _ in xs]
    m = mean(xs)
    return [(x - m) / s for x in xs]


def summarize(xs):
    return {"mean": mean(xs), "var": variance(xs), "sd": stddev(xs)}
''',
    '''
import candidate as c


def test_mean():
    assert c.mean([1, 2, 3]) == 2.0
    assert c.mean([]) == 0.0


def test_variance():
    assert abs(c.variance([1, 2, 3, 4]) - 1.6666666666666667) < 1e-9
    assert c.variance([5]) == 0.0


def test_stddev():
    assert abs(c.stddev([2, 4, 4, 4, 5, 5, 7, 9]) - 2.13808993529939) < 1e-9


def test_zscores():
    z = c.zscores([1, 2, 3])
    assert abs(sum(z)) < 1e-9
    assert c.zscores([4, 4, 4]) == [0.0, 0.0, 0.0]


def test_summarize():
    s = c.summarize([1, 2, 3])
    assert s["mean"] == 2.0
    assert abs(s["sd"] - 1.0) < 1e-9
''',
)

# --------------------------------------------------------------------------
_SEARCH = (
    '''
def lower_bound(xs, target):
    lo, hi = 0, len(xs)
    while lo < hi:
        mid = (lo + hi) // 2
        if xs[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def contains(xs, target):
    i = lower_bound(xs, target)
    return i < len(xs) and xs[i] == target


def count_range(xs, lo, hi):
    start = lower_bound(xs, lo)
    end = lower_bound(xs, hi)
    return end - start


def insert_sorted(xs, value):
    i = lower_bound(xs, value)
    return xs[:i] + [value] + xs[i:]
''',
    '''
import candidate as c


def test_lower_bound():
    assert c.lower_bound([1, 3, 5, 7], 5) == 2
    assert c.lower_bound([1, 3, 5, 7], 4) == 2
    assert c.lower_bound([], 1) == 0


def test_contains():
    assert c.contains([1, 3, 5, 7], 5)
    assert not c.contains([1, 3, 5, 7], 4)
    assert not c.contains([], 1)


def test_count_range():
    assert c.count_range([1, 2, 3, 4, 5], 2, 5) == 3
    assert c.count_range([1, 2, 3], 9, 10) == 0


def test_insert_sorted():
    assert c.insert_sorted([1, 3, 5], 4) == [1, 3, 4, 5]
    assert c.insert_sorted([], 2) == [2]
''',
)

# --------------------------------------------------------------------------
_TEXT = (
    '''
def normalize(s):
    return " ".join(s.split())


def words(s):
    return normalize(s).split(" ") if normalize(s) else []


def initials(name):
    return "".join(p[0].upper() for p in words(name))


def truncate(s, width):
    s = normalize(s)
    if len(s) <= width:
        return s
    if width <= 3:
        return s[:width]
    return s[: width - 3] + "..."


def title_case(s):
    return " ".join(w.capitalize() for w in words(s))
''',
    '''
import candidate as c


def test_normalize():
    assert c.normalize("  a   b  ") == "a b"
    assert c.normalize("") == ""


def test_words():
    assert c.words(" hello   world ") == ["hello", "world"]
    assert c.words("   ") == []


def test_initials():
    assert c.initials("ada lovelace") == "AL"
    assert c.initials("") == ""


def test_truncate():
    assert c.truncate("hello world", 8) == "hello..."
    assert c.truncate("hi", 8) == "hi"
    assert c.truncate("hello", 2) == "he"


def test_title_case():
    assert c.title_case("the  quick fox") == "The Quick Fox"
''',
)

# --------------------------------------------------------------------------
_MATRIX = (
    '''
def shape(m):
    if not m:
        return (0, 0)
    return (len(m), len(m[0]))


def transpose(m):
    r, c = shape(m)
    return [[m[i][j] for i in range(r)] for j in range(c)]


def row_sums(m):
    return [sum(row) for row in m]


def scale(m, k):
    return [[v * k for v in row] for row in m]


def trace(m):
    r, c = shape(m)
    n = min(r, c)
    return sum(m[i][i] for i in range(n))
''',
    '''
import candidate as c


def test_shape():
    assert c.shape([[1, 2], [3, 4], [5, 6]]) == (3, 2)
    assert c.shape([]) == (0, 0)


def test_transpose():
    assert c.transpose([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]
    assert c.transpose([[1, 2, 3]]) == [[1], [2], [3]]


def test_row_sums():
    assert c.row_sums([[1, 2], [3, 4]]) == [3, 7]


def test_scale():
    assert c.scale([[1, 2]], 3) == [[3, 6]]


def test_trace():
    assert c.trace([[1, 2], [3, 4]]) == 5
    assert c.trace([[1, 2, 3], [4, 5, 6]]) == 6
''',
)

# --------------------------------------------------------------------------
_INVENTORY = (
    '''
def total_units(items):
    return sum(it["qty"] for it in items)


def restock_needed(items, threshold):
    return [it["sku"] for it in items if it["qty"] < threshold]


def value_of(item):
    return item["qty"] * item["price"]


def inventory_value(items):
    return sum(value_of(it) for it in items)


def apply_discount(items, pct):
    if pct <= 0:
        return items
    out = []
    for it in items:
        copy = dict(it)
        copy["price"] = round(it["price"] * (100 - pct) / 100, 2)
        out.append(copy)
    return out
''',
    '''
import candidate as c

ITEMS = [
    {"sku": "a", "qty": 5, "price": 10.0},
    {"sku": "b", "qty": 0, "price": 4.0},
    {"sku": "d", "qty": 12, "price": 2.5},
]


def test_total_units():
    assert c.total_units(ITEMS) == 17
    assert c.total_units([]) == 0


def test_restock_needed():
    assert c.restock_needed(ITEMS, 5) == ["b"]
    assert c.restock_needed(ITEMS, 1) == ["b"]


def test_value_of():
    assert c.value_of(ITEMS[0]) == 50.0


def test_inventory_value():
    assert c.inventory_value(ITEMS) == 80.0


def test_apply_discount():
    out = c.apply_discount(ITEMS, 10)
    assert out[0]["price"] == 9.0
    assert ITEMS[0]["price"] == 10.0
    assert c.apply_discount(ITEMS, 0) is ITEMS
''',
)

# --------------------------------------------------------------------------
_DATES = (
    '''
def is_leap(year):
    if year % 400 == 0:
        return True
    if year % 100 == 0:
        return False
    return year % 4 == 0


def days_in_month(year, month):
    if month == 2:
        return 29 if is_leap(year) else 28
    if month in (4, 6, 9, 11):
        return 30
    return 31


def day_of_year(year, month, day):
    total = day
    for m in range(1, month):
        total += days_in_month(year, m)
    return total


def days_in_year(year):
    return 366 if is_leap(year) else 365
''',
    '''
import candidate as c


def test_is_leap():
    assert c.is_leap(2000)
    assert not c.is_leap(1900)
    assert c.is_leap(2024)
    assert not c.is_leap(2023)


def test_days_in_month():
    assert c.days_in_month(2024, 2) == 29
    assert c.days_in_month(2023, 2) == 28
    assert c.days_in_month(2023, 4) == 30
    assert c.days_in_month(2023, 1) == 31


def test_day_of_year():
    assert c.day_of_year(2023, 1, 1) == 1
    assert c.day_of_year(2023, 3, 1) == 60
    assert c.day_of_year(2024, 3, 1) == 61


def test_days_in_year():
    assert c.days_in_year(2024) == 366
    assert c.days_in_year(2023) == 365
''',
)

# --------------------------------------------------------------------------
_PARSING = (
    '''
def parse_kv(line):
    if "=" not in line:
        return None
    key, _, value = line.partition("=")
    return (key.strip(), value.strip())


def parse_all(text):
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        pair = parse_kv(line)
        if pair:
            out[pair[0]] = pair[1]
    return out


def coerce(value):
    if value in ("true", "false"):
        return value == "true"
    try:
        return int(value)
    except ValueError:
        return value


def build_config(text):
    raw = parse_all(text)
    return {k: coerce(v) for k, v in raw.items()}
''',
    '''
import candidate as c

TEXT = """
# comment
host = localhost
port = 8080
debug = true
name=app
"""


def test_parse_kv():
    assert c.parse_kv("a = 1") == ("a", "1")
    assert c.parse_kv("nokey") is None


def test_parse_all():
    d = c.parse_all(TEXT)
    assert d["host"] == "localhost"
    assert d["port"] == "8080"
    assert "# comment" not in d


def test_coerce():
    assert c.coerce("12") == 12
    assert c.coerce("true") is True
    assert c.coerce("abc") == "abc"


def test_build_config():
    cfg = c.build_config(TEXT)
    assert cfg["port"] == 8080
    assert cfg["debug"] is True
    assert cfg["name"] == "app"
''',
)

# --------------------------------------------------------------------------
_GEOMETRY = (
    '''
def distance(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def perimeter(points):
    if len(points) < 2:
        return 0.0
    total = 0.0
    for i in range(len(points)):
        total += distance(points[i], points[(i + 1) % len(points)])
    return total


def centroid(points):
    if not points:
        return (0.0, 0.0)
    n = len(points)
    return (sum(p[0] for p in points) / n, sum(p[1] for p in points) / n)


def bounding_box(points):
    if not points:
        return (0, 0, 0, 0)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))
''',
    '''
import candidate as c

SQ = [(0, 0), (0, 2), (2, 2), (2, 0)]


def test_distance():
    assert c.distance((0, 0), (3, 4)) == 5.0
    assert c.distance((1, 1), (1, 1)) == 0.0


def test_perimeter():
    assert abs(c.perimeter(SQ) - 8.0) < 1e-9
    assert c.perimeter([(0, 0)]) == 0.0


def test_centroid():
    assert c.centroid(SQ) == (1.0, 1.0)
    assert c.centroid([]) == (0.0, 0.0)


def test_bounding_box():
    assert c.bounding_box(SQ) == (0, 0, 2, 2)
''',
)

# --------------------------------------------------------------------------
_STACK = (
    '''
def make_stack():
    return []


def push(stack, value):
    stack.append(value)
    return stack


def pop(stack):
    if not stack:
        return None
    return stack.pop()


def peek(stack):
    if not stack:
        return None
    return stack[-1]


def is_balanced(text):
    pairs = {"(": ")", "[": "]", "{": "}"}
    s = []
    for ch in text:
        if ch in pairs:
            s.append(pairs[ch])
        elif ch in pairs.values():
            if not s or s[-1] != ch:
                return False
            s.pop()
    return len(s) == 0
''',
    '''
import candidate as c


def test_make_stack():
    assert c.make_stack() == []


def test_push_pop():
    s = c.make_stack()
    c.push(s, 1)
    c.push(s, 2)
    assert c.pop(s) == 2
    assert c.pop(s) == 1
    assert c.pop(s) is None


def test_peek():
    s = c.make_stack()
    assert c.peek(s) is None
    c.push(s, 42)
    assert c.peek(s) == 42
    assert len(s) == 1


def test_balanced():
    assert c.is_balanced("([]){}")
    assert not c.is_balanced("([)]")
    assert c.is_balanced("")
    assert not c.is_balanced("((")
''',
)

# --------------------------------------------------------------------------
_GRAPH = (
    '''
def make_graph():
    return {}


def add_edge(g, u, v):
    g.setdefault(u, []).append(v)
    g.setdefault(v, []).append(u)
    return g


def neighbors(g, node):
    return g.get(node, [])


def bfs(g, start):
    visited = []
    queue = [start]
    seen = {start}
    while queue:
        node = queue.pop(0)
        visited.append(node)
        for nb in neighbors(g, node):
            if nb not in seen:
                seen.add(nb)
                queue.append(nb)
    return visited


def has_path(g, start, end):
    return end in bfs(g, start)
''',
    '''
import candidate as c


def test_make_graph():
    assert c.make_graph() == {}


def test_add_edge():
    g = c.make_graph()
    c.add_edge(g, "a", "b")
    assert "b" in c.neighbors(g, "a")
    assert "a" in c.neighbors(g, "b")


def test_bfs():
    g = c.make_graph()
    c.add_edge(g, 1, 2)
    c.add_edge(g, 2, 3)
    c.add_edge(g, 1, 4)
    result = c.bfs(g, 1)
    assert result[0] == 1
    assert set(result) == {1, 2, 3, 4}


def test_has_path():
    g = c.make_graph()
    c.add_edge(g, "a", "b")
    c.add_edge(g, "b", "c")
    assert c.has_path(g, "a", "c")
    assert not c.has_path(g, "a", "z")
''',
)

# --------------------------------------------------------------------------
_INTERVAL = (
    '''
def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def merge(a, b):
    if not overlaps(a, b):
        return None
    return (min(a[0], b[0]), max(a[1], b[1]))


def contains(interval, point):
    return interval[0] <= point < interval[1]


def length(interval):
    return max(0, interval[1] - interval[0])


def merge_all(intervals):
    if not intervals:
        return []
    s = sorted(intervals, key=lambda x: x[0])
    merged = [s[0]]
    for iv in s[1:]:
        if overlaps(merged[-1], iv):
            merged[-1] = merge(merged[-1], iv)
        else:
            merged.append(iv)
    return merged
''',
    '''
import candidate as c


def test_overlaps():
    assert c.overlaps((1, 5), (3, 7))
    assert not c.overlaps((1, 3), (5, 7))
    assert c.overlaps((1, 5), (5, 7)) is False


def test_merge():
    assert c.merge((1, 5), (3, 7)) == (1, 7)
    assert c.merge((1, 3), (5, 7)) is None


def test_contains():
    assert c.contains((1, 5), 3)
    assert not c.contains((1, 5), 5)
    assert c.contains((1, 5), 1)


def test_length():
    assert c.length((2, 7)) == 5
    assert c.length((5, 3)) == 0


def test_merge_all():
    assert c.merge_all([(1, 3), (2, 5), (8, 10)]) == [(1, 5), (8, 10)]
    assert c.merge_all([]) == []
    assert c.merge_all([(1, 10), (2, 3)]) == [(1, 10)]
''',
)

# --------------------------------------------------------------------------
_CACHE = (
    '''
def make_cache(capacity):
    return {"cap": capacity, "keys": [], "store": {}}


def cache_get(cache, key):
    if key not in cache["store"]:
        return None
    cache["keys"].remove(key)
    cache["keys"].append(key)
    return cache["store"][key]


def cache_put(cache, key, value):
    if key in cache["store"]:
        cache["keys"].remove(key)
    elif len(cache["keys"]) >= cache["cap"]:
        evicted = cache["keys"].pop(0)
        del cache["store"][evicted]
    cache["keys"].append(key)
    cache["store"][key] = value


def cache_size(cache):
    return len(cache["store"])


def cache_keys(cache):
    return list(cache["keys"])
''',
    '''
import candidate as c


def test_make_cache():
    ca = c.make_cache(2)
    assert c.cache_size(ca) == 0


def test_put_get():
    ca = c.make_cache(2)
    c.cache_put(ca, "a", 1)
    assert c.cache_get(ca, "a") == 1
    assert c.cache_get(ca, "b") is None


def test_eviction():
    ca = c.make_cache(2)
    c.cache_put(ca, "a", 1)
    c.cache_put(ca, "b", 2)
    c.cache_put(ca, "c", 3)
    assert c.cache_get(ca, "a") is None
    assert c.cache_get(ca, "c") == 3
    assert c.cache_size(ca) == 2


def test_lru_order():
    ca = c.make_cache(2)
    c.cache_put(ca, "a", 1)
    c.cache_put(ca, "b", 2)
    c.cache_get(ca, "a")
    c.cache_put(ca, "c", 3)
    assert c.cache_get(ca, "b") is None
    assert c.cache_get(ca, "a") == 1


def test_cache_keys():
    ca = c.make_cache(3)
    c.cache_put(ca, "x", 10)
    c.cache_put(ca, "y", 20)
    assert c.cache_keys(ca) == ["x", "y"]
''',
)


# --------------------------------------------------------------------------
_SORTING = (
    '''
def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def merge_sort(xs):
    if len(xs) <= 1:
        return xs
    mid = len(xs) // 2
    return merge(merge_sort(xs[:mid]), merge_sort(xs[mid:]))


def is_sorted(xs):
    return all(xs[i] <= xs[i + 1] for i in range(len(xs) - 1))


def rank(xs):
    """Return the rank (0-based position in sorted order) for each element."""
    sorted_xs = merge_sort(list(xs))
    return [sorted_xs.index(x) for x in xs]
''',
    '''
import candidate as c


def test_merge():
    assert c.merge([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]
    assert c.merge([], [1, 2]) == [1, 2]
    assert c.merge([1, 2], []) == [1, 2]


def test_merge_sort():
    assert c.merge_sort([3, 1, 4, 1, 5, 9, 2, 6]) == [1, 1, 2, 3, 4, 5, 6, 9]
    assert c.merge_sort([]) == []
    assert c.merge_sort([7]) == [7]


def test_is_sorted():
    assert c.is_sorted([1, 2, 3])
    assert c.is_sorted([])
    assert not c.is_sorted([3, 1, 2])
    assert c.is_sorted([5, 5, 5])


def test_rank():
    assert c.rank([3, 1, 2]) == [2, 0, 1]
    assert c.rank([10]) == [0]
''',
)

# --------------------------------------------------------------------------
_ENCODING = (
    '''
def caesar_encrypt(text, shift):
    result = []
    for ch in text:
        if ch.isalpha():
            base = ord('A') if ch.isupper() else ord('a')
            result.append(chr((ord(ch) - base + shift) % 26 + base))
        else:
            result.append(ch)
    return ''.join(result)


def caesar_decrypt(text, shift):
    return caesar_encrypt(text, -shift)


def rle_encode(s):
    if not s:
        return []
    out = []
    count = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            count += 1
        else:
            out.append((s[i - 1], count))
            count = 1
    out.append((s[-1], count))
    return out


def rle_decode(encoded):
    return ''.join(ch * n for ch, n in encoded)


def rle_compress_ratio(s):
    """Ratio of encoded pairs to original length; lower is better compression."""
    if not s:
        return 0.0
    encoded = rle_encode(s)
    return len(encoded) / len(s)
''',
    '''
import candidate as c


def test_caesar_encrypt():
    assert c.caesar_encrypt('abc', 3) == 'def'
    assert c.caesar_encrypt('xyz', 3) == 'abc'
    assert c.caesar_encrypt('Hello, World!', 13) == 'Uryyb, Jbeyq!'
    assert c.caesar_encrypt('', 5) == ''


def test_caesar_decrypt():
    assert c.caesar_decrypt('def', 3) == 'abc'
    assert c.caesar_decrypt(c.caesar_encrypt('Secret', 7), 7) == 'Secret'


def test_rle_encode():
    assert c.rle_encode('aaabbc') == [('a', 3), ('b', 2), ('c', 1)]
    assert c.rle_encode('') == []
    assert c.rle_encode('a') == [('a', 1)]


def test_rle_decode():
    assert c.rle_decode([('a', 3), ('b', 2), ('c', 1)]) == 'aaabbc'
    assert c.rle_decode([]) == ''


def test_rle_roundtrip():
    for s in ['aabbcc', 'abcd', 'aaaa']:
        assert c.rle_decode(c.rle_encode(s)) == s


def test_rle_compress_ratio():
    assert c.rle_compress_ratio('') == 0.0
    assert c.rle_compress_ratio('aaaa') < 1.0
    assert c.rle_compress_ratio('abcd') == 1.0
''',
)


REFERENCE_PROGRAMS: dict[str, tuple[str, str]] = {
    "stats": _STATS,
    "search": _SEARCH,
    "text": _TEXT,
    "matrix": _MATRIX,
    "inventory": _INVENTORY,
    "dates": _DATES,
    "parsing": _PARSING,
    "geometry": _GEOMETRY,
    "stack": _STACK,
    "graph": _GRAPH,
    "interval": _INTERVAL,
    "cache": _CACHE,
    "sorting": _SORTING,
    "encoding": _ENCODING,
}
