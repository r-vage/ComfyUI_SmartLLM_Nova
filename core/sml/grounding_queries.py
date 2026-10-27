# Independent phrase-grounding queries share a loaded model and one image output.

import re

from .vlm_detection import scale_normalized_detection_to_pixels


def split_grounding_queries(task: str, text: str) -> list[str]:
    # Other tasks can contain prose, commas or document questions.
    if task != "Caption to Phrase Grounding":
        return [text]
    queries = []
    seen = set()
    for part in re.split(r"[,;\r\n]+", text):
        phrase = part.strip()
        if phrase and phrase.casefold() not in seen:
            queries.append(phrase)
            seen.add(phrase.casefold())
    return queries or [text]


def merge_grounding_results(
    results: list[dict], image_size: tuple[int, int]
) -> tuple[dict, list[int] | None]:
    # Preserve the single-query contract. Multiple queries can return different
    # coordinate spaces, so convert each before concatenating the aligned fields.
    if len(results) == 1:
        return results[0], None
    merged = dict(results[0]) if results else {}
    merged.update(bboxes=[], labels=[], coord_range=0)
    defaults = {"confidences": 1.0, "quad_boxes": [], "polygons": []}
    fields = {key: default for key, default in defaults.items() if any(key in item for item in results)}
    for key in fields:
        merged[key] = []
    groups = []
    for query_index, result in enumerate(results):
        data = scale_normalized_detection_to_pixels(result, image_size)
        boxes = data.get("bboxes", [])
        labels = data.get("labels", [])
        merged["bboxes"].extend(boxes)
        merged["labels"].extend(labels[index] if index < len(labels) else "object" for index in range(len(boxes)))
        groups.extend([query_index] * len(boxes))
        for key, default in fields.items():
            values = data.get(key, [])
            merged[key].extend(values[index] if index < len(values) else default for index in range(len(boxes)))
    return merged, groups
