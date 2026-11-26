"""Conservative frame-wide scale evidence for a single observed shot."""
import json
import re
from math import atan2, degrees, exp, hypot, log

import cv2
import numpy as np

from .scene_motion import parse_scene_response


def _features(frame, detector):
    pixels = frame.pixels.detach().float().cpu().numpy()
    height, width = pixels.shape[:2]
    factor = min(1.0, 384 / max(width, height))
    if factor < 1:
        pixels = cv2.resize(pixels, (max(1, round(width * factor)), max(1, round(height * factor))),
                            interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(pixels[..., :3], cv2.COLOR_RGB2GRAY)
    gray = np.uint8(np.clip(gray * 255, 0, 255))
    keys, desc = detector.detectAndCompute(gray, None)
    points = np.float32([key.pt for key in keys]).reshape(-1, 2)
    return points, desc, np.array(gray.shape[::-1])


def _reliable_matches(source, target, matcher):
    matches = set()
    for pair in matcher.knnMatch(source, target, k=2):
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
            matches.add((pair[0].queryIdx, pair[0].trainIdx))
    return matches


def _scale_step(first, second, matcher):
    points, features, size = first
    other, descriptors, other_size = second
    if features is None or descriptors is None or len(descriptors) < 2 or not np.array_equal(size, other_size):
        return None
    # Reciprocal matches exclude repeated details and give zoom-in and zoom-out the same support.
    reverse = {(j, i) for i, j in _reliable_matches(descriptors, features, matcher)}
    pairs = sorted(_reliable_matches(features, descriptors, matcher) & reverse)
    if len(pairs) < 16:
        return None
    a = points[[i for i, _ in pairs]]
    b = other[[j for _, j in pairs]]
    transform, mask = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2, maxIters=300)
    if transform is None or mask is None or mask.mean() < .70:
        return None
    inliers = mask.ravel().astype(bool)
    # The fitted movement must cover opposing outer regions in both images.
    # Matches restricted to the performer's face or hands cannot establish a change in framing.
    for positions in (a[inliers], b[inliers]):
        positions = positions / size
        if np.any(positions.min(axis=0) > .30) or np.any(positions.max(axis=0) < .70):
            return None
        if cv2.contourArea(cv2.convexHull(positions.astype(np.float32))) < .22:
            return None
    scale = hypot(transform[0, 0], transform[1, 0])
    if not .60 < scale < 1.65 or abs(degrees(atan2(transform[1, 0], transform[0, 0]))) > 12:
        return None
    return log(scale)


def measured_shot_scale(frames):
    """Report sustained magnification across the frame. Skip cuts, sparse matches and single jumps."""
    if len(frames) < 3 or frames[0].shot != frames[-1].shot:
        return None
    selected = np.linspace(0, len(frames) - 1, min(8, len(frames)), dtype=int)
    detector = cv2.ORB_create(nfeatures=750, edgeThreshold=12, fastThreshold=10)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    changes = []
    try:
        previous = _features(frames[selected[0]], detector)
        for index in selected[1:]:
            current = _features(frames[index], detector)
            change = _scale_step(previous, current, matcher)
            if change is None:
                return None
            changes.append(change)
            previous = current
    except cv2.error:
        # Camera geometry is optional evidence. A usable scene can still be described without it.
        return None
    total = sum(changes)
    sign = 1 if total >= 0 else -1
    if (abs(total) < log(1.12) or sum(change * sign > .012 for change in changes) < 2
            or min(change * sign for change in changes) < -.025 or max(map(abs, changes)) > .80 * abs(total)):
        return None
    return {"framing": "tightens" if total > 0 else "widens", "scale_ratio": round(exp(total), 2)}


# Scene-wide scale establishes a change of framing. It cannot distinguish a lens zoom
# from a crop or a moving camera, so the fallback describes the visible result.
_FIXED_CAMERA = re.compile(
    r"\b(?:the )?camera\s+(?:(?:is|stays?|remains?|appears?)\s+)?(?:static|fixed|stationary|locked[- ]off)"
    r"(?:\s+throughout(?: the shot)?)?\b|\b(?:from (?:a |the )?)?(?:static|fixed|stationary|locked[- ]off) camera\b|"
    r"\b(?:static|fixed|locked[- ]off)\s+(?=(?:medium|wide|close|full|long|extreme|shot)\b)|"
    r"\b(?:no|without) (?:visible |apparent )?camera (?:motion|movement)\b", re.IGNORECASE,
)
_FIXED_FIELD = re.compile(r"^(?:static|fixed|stationary|locked[- ]off|none|unknown|not visible|n/a)[.! ]*$", re.IGNORECASE)
_SCALE_LANGUAGE = {
    "tightens": re.compile(r"\bzoom(?:s|ing|ed)?[- ]in\b|\b(?:fram(?:e|ing)|shot) (?:\w+ly )?tightens?\b|"
                          r"\btightens? (?:to|into) (?:an? |the )?(?:extreme |medium )?close[- ]up\b", re.IGNORECASE),
    "widens": re.compile(r"\bzoom(?:s|ing|ed)?[- ]out\b|\b(?:fram(?:e|ing)|shot) (?:\w+ly )?widens?\b|"
                         r"\bwidens? (?:to|into) (?:an? |the )?(?:[\w-]+ ){0,2}shot\b", re.IGNORECASE),
}


def _aligned_direction(text, direction):
    # When the model reverses a zoom, its claimed destination is unreliable too.
    # Stop before the performer clause so the observed gesture survives.
    opposite = "out" if direction == "tightens" else "in"
    pattern = (rf"\b(?:(?:the )?camera\s+)?(?:(?:slowly|gradually|quickly) )?zoom(?:s|ing|ed)?[- ]{opposite}\b"
               r"(?:\s+(?:to|into) (?:an? |the )?(?:[\w-]+ ){0,3}(?:shot|close[- ]up))?")
    def replace(match):
        prefix = "The" if match.start() == 0 or match[0][:1].isupper() else "the"
        return f"{prefix} framing progressively {direction}"
    return re.sub(pattern, replace, text, flags=re.IGNORECASE)


def _clear_fixed_claim(text):
    text = _FIXED_CAMERA.sub("", text)
    text = re.sub(r"\s+([.,;])", r"\1", text)
    text = re.sub(r"([.,;])(?:\s*[,;.])+", r"\1", text)
    return text.strip(" ,;")


def reconcile_camera_response(text, scale):
    """Ground a generated scene paragraph and keep the model's wording for the input evidence."""
    if not scale:
        return text, {}
    try:
        value = parse_scene_response(text)
    except (TypeError, ValueError):
        return text, {}
    if not isinstance(value, dict):
        return text, {}
    direction = scale["framing"]
    observed = f"The framing progressively {direction}."
    original = {}
    for field in ("camera_motion", "performance_prompt"):
        wording = value.get(field)
        if not isinstance(wording, str) or not wording.strip():
            continue  # An incomplete response still belongs to the normal recovery path.
        cleaned = _clear_fixed_claim(wording).strip()
        if field == "camera_motion" and _FIXED_FIELD.fullmatch(cleaned):
            cleaned = ""
        if not _SCALE_LANGUAGE[direction].search(cleaned):
            cleaned = _aligned_direction(cleaned, direction)
        if not _SCALE_LANGUAGE[direction].search(cleaned):
            cleaned = (cleaned.rstrip(" .,;") + ". " if cleaned.strip(" .,;") else "") + observed
        if cleaned != wording:
            original[field] = wording
            value[field] = cleaned
    return (json.dumps(value, ensure_ascii=False), {"model_camera_wording": original}) if original else (text, {})
