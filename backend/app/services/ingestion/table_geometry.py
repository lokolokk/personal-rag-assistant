"""Conservative reconstruction of ruled numeric tables from source word geometry."""

import re
import statistics
from typing import Any


def _rows(words: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    rows: list[list[dict[str, Any]]] = []
    for word in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if not rows or abs(word["top"] - rows[-1][0]["top"]) > 2:
            rows.append([])
        rows[-1].append(word)
    return [sorted(row, key=lambda w: w["x0"]) for row in rows]


def ruled_matrix(
    page: Any,
    bbox: tuple[float, float, float, float],
    rules: list[dict[str, float]],
) -> list[list[str]] | None:
    """Use a full-width header rule and aligned numeric rows, or keep the old fallback.

    No vocabulary identifies a paper or metric. Parent labels are assigned only
    through nested rule spans; unruled top-tier labels use child-group centers.
    Missing row labels remain blank unless a horizontal band proves a shared label.
    """
    x0, top, x1, bottom = bbox
    words: list[dict[str, Any]] = page.crop(bbox).extract_words(y_tolerance=2)
    full = sorted(
        {
            r["top"]
            for r in rules
            if top <= r["top"] <= bottom and abs(r["x0"] - x0) < 5 and abs(r["x1"] - x1) < 5
        }
    )
    if len(full) < 3:
        return None
    header_end = full[1]
    header_rows = _rows([w for w in words if w["bottom"] < header_end])
    if not header_rows:
        return None
    leaves = header_rows[-1]
    body = [w for w in words if w["top"] > header_end and w["bottom"] < bottom]
    if len(leaves) < 4 or not body:
        return None
    # Only admit columns whose source header sits above repeated numeric entries.
    numeric = re.compile(r"^[+−-]?\d+(?:\.\d+)?%?$")
    centers = [(w["x0"] + w["x1"]) / 2 for w in leaves]
    metric_start = next(
        (
            i
            for i, center in enumerate(centers)
            if sum(
                bool(numeric.fullmatch(w["text"])) and abs((w["x0"] + w["x1"]) / 2 - center) < 5
                for w in body
            )
            >= 2
        ),
        len(leaves),
    )
    if metric_start < 1 or len(leaves) - metric_start < 3:
        return None
    metric_centers = centers[metric_start:]
    # Numeric metric columns must have clear, ordered separation.
    gaps = [b - a for a, b in zip(metric_centers, metric_centers[1:], strict=False)]
    if min(gaps) < 8:
        return None
    first_boundary = metric_centers[0] - gaps[0] / 2
    boundaries = (
        [x0]
        + [
            (a + b) / 2
            for a, b in zip(centers[:metric_start], centers[1:metric_start], strict=False)
        ]
        + [first_boundary]
        + [(a + b) / 2 for a, b in zip(metric_centers, metric_centers[1:], strict=False)]
        + [x1]
    )

    headers = [w["text"] for w in leaves]
    child_centers = metric_centers
    for tier in reversed(header_rows[:-1]):
        # Horizontal strokes directly below the tier provide explicit group spans.
        tier_bottom = max(w["bottom"] for w in tier)
        tier_top = min(w["top"] for w in tier)
        spans = [
            r
            for r in rules
            if tier_bottom <= r["top"] <= tier_bottom + 5
            and r["x0"] > first_boundary
            and r["x1"] < x1
        ]
        tier_centers = [(w["x0"] + w["x1"]) / 2 for w in tier]
        assigned: list[str] = []
        for center in child_centers:
            eligible = [r for r in spans if r["x0"] <= center <= r["x1"]]
            if spans:
                if len(eligible) != 1:
                    return None
                candidates = [
                    w
                    for w in tier
                    if eligible[0]["x0"] <= (w["x0"] + w["x1"]) / 2 <= eligible[0]["x1"]
                ]
                if len(candidates) != 1:
                    return None
                assigned.append(candidates[0]["text"])
            else:
                # An upper tier must partition already established child groups.
                if len(tier) < 2 or len(tier) >= len(set(headers[metric_start:])):
                    return None
                nearest = min(range(len(tier)), key=lambda i: abs(tier_centers[i] - center))
                distances = sorted(abs(value - center) for value in tier_centers)
                if distances[1] - distances[0] < 2:
                    return None
                assigned.append(tier[nearest]["text"])
        if set(assigned) != {w["text"] for w in tier}:
            return None
        for i, label in enumerate(assigned, metric_start):
            headers[i] = label + " / " + headers[i]
        # Top parents are centered over the subordinate group, not each metric.
        child_centers = []
        for i in range(metric_start, len(headers)):
            label = assigned[i - metric_start]
            members = [metric_centers[j] for j, value in enumerate(assigned) if value == label]
            child_centers.append(statistics.mean(members))
        if tier_top < top:
            return None

    first_column = [
        w for w in body if first_boundary <= (w["x0"] + w["x1"]) / 2 < boundaries[metric_start + 1]
    ]
    baselines = _rows(first_column)
    if len(baselines) < 2 or any(
        len(row) != 1 or not numeric.fullmatch(row[0]["text"]) for row in baselines
    ):
        return None
    y_values = [row[0]["top"] for row in baselines]
    height = statistics.median(w["bottom"] - w["top"] for row in baselines for w in row)
    matrix = [headers]
    used_words: set[int] = set()
    for y in y_values:
        cells: list[str] = []
        for col, (left, right) in enumerate(zip(boundaries, boundaries[1:], strict=False)):
            column_words = [w for w in body if left <= (w["x0"] + w["x1"]) / 2 < right]
            if col == 0 and metric_start > 1:
                # The outer label can span several rows within a ruled band.
                band_top = max(v for v in full if v < y)
                band_bottom = min(v for v in full if v > y)
                band_words = [w for w in column_words if band_top < w["top"] < band_bottom]
                band_rows = [v for v in y_values if band_top < v < band_bottom]
                centered = (
                    bool(band_words)
                    and abs(
                        (min(w["top"] for w in band_words) + max(w["bottom"] for w in band_words))
                        / 2
                        - (band_top + band_bottom) / 2
                    )
                    < height / 2
                )
                shared = centered and len(_rows(band_words)) < len(band_rows)
                selected = (
                    band_words if shared else [w for w in column_words if abs(w["top"] - y) < 2]
                )
            else:
                selected = [w for w in column_words if abs(w["top"] - y) < height * 0.6]
            selected.sort(key=lambda w: (w["x0"], w["top"]))
            used_words.update(id(word) for word in selected)
            parts: list[str] = []
            for word in selected:
                small = word["bottom"] - word["top"] < height * 0.8
                if col >= metric_start and small and word["top"] > y + 1 and parts:
                    parts[-1] += "_{" + word["text"] + "}"
                else:
                    parts.append(word["text"])
            if col >= metric_start and (not parts or len(parts) != 1):
                return None
            cells.append(" ".join(parts))
        matrix.append(cells)
    if len(used_words) != len(body):
        # Wrapped labels or unassigned annotations must not silently disappear.
        return None
    return matrix


def following_caption(page: Any, bbox: tuple[float, float, float, float]) -> str:
    """Copy a nearby, aligned numbered caption verbatim; never infer table identity."""
    x0, _, x1, bottom = bbox
    words = page.crop(
        (max(0, x0 - 2), bottom, min(page.width, x1 + 2), min(page.height, bottom + 120))
    ).extract_words(y_tolerance=2)
    rows = _rows(words)
    if not rows or rows[0][0]["top"] - bottom > 20:
        return ""
    first = " ".join(w["text"] for w in rows[0])
    if not re.match(r"^Table\s+\d+\s*:", first, re.IGNORECASE):
        return ""
    result = [first]
    last_bottom = max(w["bottom"] for w in rows[0])
    for row in rows[1:8]:
        if min(w["top"] for w in row) - last_bottom > 5:
            break
        result.append(" ".join(w["text"] for w in row))
        last_bottom = max(w["bottom"] for w in row)
    return " ".join(result)
