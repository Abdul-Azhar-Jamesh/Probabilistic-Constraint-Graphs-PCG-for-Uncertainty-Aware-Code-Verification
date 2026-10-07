"""Intentionally faulty example for graph diagnosis and generated contracts."""


def normalize(values: list[int]) -> list[int]:
    ordered = sorted(values)
    truncated = ordered[:-1]  # Violates the permutation contract.
    return truncated


def ratio(total: float, count: int) -> float:
    if count == 0:
        raise ValueError("empty input")
    divisor = count - count  # Earlier origin of the later runtime failure.
    _unrelated = 123
    numerator = total
    return numerator / divisor
