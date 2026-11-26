"""Analyze bounded visual windows independently, then combine their evidence."""
import json
from contextlib import closing

import torch

from .media_inputs import VisualFrame, check_interrupt, pack_grid, visual_windows
from .scene_camera import measured_shot_scale, reconcile_camera_response
from .scene_motion import (
    SCENE_TASK_MODES,
    black_scene_observation,
    encode_scene_evidence,
    has_recovered_scene_text,
    parse_scene_response,
    scene_context_ids,
    scene_contract_issue,
    scene_request_prompt,
)


def _context(prompt):
    raw = (prompt or "").lstrip()
    try:
        value, end = json.JSONDecoder().raw_decode(raw)
        if isinstance(value, dict):
            return value, raw[end:]
    except ValueError:
        pass
    return {"scene_id": 0, "window_id": 0, "source_context": prompt or ""}, ""


def _window_prompt(prompt, frames, batch, layout, window_number, frame_scale=None):
    # Reuse positional mapping for Eclipse disk jobs. The original scene/window
    # IDs stay intact; internal window numbers are separate metadata.
    positions = tuple(frame.position for frame in frames)
    context, suffix = _context(scene_request_prompt(prompt, batch, positions))
    context["visual_window"] = {"index": window_number, "shot": frames[0].shot,
                                 "positions": list(positions),
                                 "seconds": [frame.seconds for frame in frames]}
    if layout:
        context["request_images"] = {
            "count": 1, "frame_count": len(frames), "positions": list(positions),
            "presentation": "one contact sheet, read numbered tiles left to right, top to bottom",
            "first_tile": 1, "last_tile": len(frames), "layout": layout,
        }
        instruction = (
            "The image is a chronological contact sheet from one supplied window. Each numbered tile is a separate "
            "moment, not a different person or a collage in the scene. Read tiles left to right, top to bottom. "
            f"Read the starting pose from tile 1 and the ending pose from tile {len(frames)}. "
            "Ignore the artificial gutters and labels in your description. Describe only the visible action; "
            "do not write one caption per tile."
        )
    else:
        instruction = "The images are chronological samples from one supplied window; describe only the visible action."
    instruction += (
        " This may be an excerpt of a longer scene and may contain an undetected edit. "
        "Do not invent movement connecting an abrupt pose or framing change. "
        "Do not invent a new beginning or a return to the starting pose. "
        "Retain relevant framing, posture, movements, expressions and task-appropriate scene detail. "
        "Respect the selected task's requested format and level of detail. "
        "Summarize repeated motion once; do not infer motion or audio from gaps."
    )
    return _attach_scale_prompt(json.dumps(context, ensure_ascii=False) + suffix + "\n" + instruction, frame_scale)


def _attach_scale_prompt(prompt, scale):
    if not scale:
        return prompt
    context, suffix = _context(prompt)
    context["frame_scale_evidence"] = scale
    instruction = (
        f" Matching features across the frame show that the framing progressively {scale['framing']} "
        f"(image scale ends at {scale['scale_ratio']} times its starting size). "
        "Describe this change in camera_motion and performance_prompt. Use the samples to identify the opening "
        "and ending shot sizes when visible. The measurement does not establish how the shot was filmed."
    )
    return json.dumps(context, ensure_ascii=False) + suffix + instruction


def direct_scene_camera(prompt, image, positions, task, omitted):
    """Use the same shot measurement for a directly submitted multi-image request."""
    scale = None
    if (SCENE_TASK_MODES.get(task) == "motion" and len(image) >= 3 and not omitted
            and not _context(prompt)[0].get("excluded_source_cuts")):
        frames = [VisualFrame(pixels, position) for pixels, position in zip(image, positions, strict=True)]
        scale = measured_shot_scale(frames)
    return _attach_scale_prompt(prompt, scale), scale


def _merge_payload(observation, structured):
    if not structured:
        return observation
    value = parse_scene_response(observation["response"])
    # Intermediate boundary poses are snapshots, not extra actions. A facts-led
    # merge avoids restarting each window by copying its opening/final clauses.
    fields = ("framing", "camera_motion", "appearance", "actions", "gaze_expression", "uncertainties")
    return {key: observation[key] for key in ("shot", "last_shot", "start", "end")} | {
        "observations": {key: value[key] for key in fields if key in value},
        **({"frame_scale_evidence": observation["frame_scale_evidence"]} if "frame_scale_evidence" in observation else {}),
        # A T2V response may omit the optional appearance audit. Its paragraph
        # is then the only source of appearance/setting detail for synthesis.
        **({"additional_description": value.get("performance_prompt")}
           if value.get("description_mode") == "full" and not value.get("appearance") else {}),
    }


def _boundary_evidence(group):
    """Only the outer snapshots are scene boundaries; overlap cannot reset them."""
    first = parse_scene_response(group[0]["response"])
    last = parse_scene_response(group[-1]["response"])
    return {"opening_framing": first.get("framing"), "scene_start_pose": first.get("start_pose"),
            "scene_end_pose": last.get("end_pose")}


def _join_scene_windows(observations, task, reason):
    """Keep the original, validated windows in chronology when synthesis cannot finish."""
    records = [parse_scene_response(item["response"]) for item in observations]
    first, last = records[0], records[-1]
    result = {key: first[key] for key in ("scene_id", "window_id", "framing", "start_pose")}
    result.update(end_pose=last["end_pose"], description_mode=SCENE_TASK_MODES[task])
    for key in ("actions", "gaze_expression", "uncertainties", "review_warnings"):
        result[key] = list(dict.fromkeys(text for r in records for text in (r.get(key) or [])
                                         if isinstance(text, str)))
    result["review_warnings"].append(reason)
    for key in ("camera_motion", "appearance"):
        values = [r[key] for r in records if isinstance(r.get(key), str) and r[key].strip()]
        if values:
            result[key] = " ".join(dict.fromkeys(values))
    paragraphs = []
    previous = None
    for window, record in zip(observations, records, strict=True):
        paragraph = record["performance_prompt"].strip()
        overlaps = previous and window["shot"] == previous["last_shot"] and window["start"] <= previous["end"]
        if not overlaps or paragraph != paragraphs[-1]:
            paragraphs.append(paragraph)
        previous = window
    result["performance_prompt"] = " ".join(paragraphs)
    return json.dumps(result, ensure_ascii=False)


def _merge(observations, prompt, task, generate, context_size, max_tokens):
    """Hierarchical text-only consolidation; never feed summaries into vision."""
    context, suffix = _context(prompt)
    structured = task in SCENE_TASK_MODES
    if structured:
        expected = scene_context_ids(prompt)
        recovered = False
        for window in observations:
            text = window["response"]
            if scene_contract_issue(text, expected, task, check_content=False):
                # Return the failed window for the collector's diagnostic. An
                # invalid or misidentified reply is never evidence for another call.
                return text
            if has_recovered_scene_text(parse_scene_response(text)):
                recovered = True
        if recovered:
            return _join_scene_windows(observations, task, "A window was incomplete. Its retained text is included in scene order.")
    context.pop("request_images", None)
    context.pop("frame_indices", None)
    context.pop("frame_seconds", None)
    instruction = (
        "Combine the following independently observed chronological windows into ONE response in the selected task's format. "
        "Use only these observations; images are not attached to this synthesis request. Overlapping windows show some of "
        "the same action: their shared source positions are shared evidence, not another action cycle. "
        "Combine descriptions of the same movement phase instead of concatenating each window action list. "
        "Preserve separately observed repetitions and distinct intermediate gestures, "
        "framing, camera movement, expressions and all task-appropriate appearance/setting detail. Keep simultaneous actions together. "
        "Start the generation paragraph with opening_framing and scene_start_pose when supplied, and preserve scene_end_pose "
        "as the ending snapshot. Keep observed framing changes after the opening. Carry the observed camera angle "
        "and within-shot camera path into the paragraph, including the direction and endpoint of a zoom. "
        "Preserve a change of shot size without naming a camera move when its cause is uncertain. "
        "Intermediate window boundaries do not imply a reset or another gesture. "
        "An unchanged shot number does not prove continuous motion: retain reported discontinuities and "
        "uncertainties without inventing connecting movement. "
        "Cover the whole observation sequence, not only its final window. A changed shot number is a cut: preserve the changed framing/pose "
        "without inventing motion across the cut. Do not copy window labels, timestamps or analysis instructions into the prompt. "
        "Follow the selected task's requested length and level of detail for the generation paragraph. "
        "Do not add unsupported facts. Return only the selected task's final output."
    )
    base = json.dumps(context, ensure_ascii=False) + suffix + "\n" + instruction
    # Conservative character allowance; leave room for the task system, answer,
    # and tokenization overhead. Never silently truncate an observation.
    budget = max(2048, (context_size - min(max_tokens, context_size // 2)) * 2 - len(base) - 4000)
    current = observations
    while len(current) > 1:
        merged = []
        index = 0
        while index < len(current):
            group, payloads, size = [], [], 0
            while index < len(current):
                payload = _merge_payload(current[index], structured)
                encoded = json.dumps(payload, ensure_ascii=False)
                if group and size + len(encoded) > budget:
                    break
                group.append(current[index])
                payloads.append(payload)
                size += len(encoded)
                index += 1
            if len(group) == 1:
                merged.extend(group)
                continue
            check_interrupt()
            boundary = _boundary_evidence(group) if structured else {}
            evidence = {**boundary, "windows": payloads}
            text, _ = generate(base + "\nObserved evidence:\n" + json.dumps(evidence, ensure_ascii=False),
                               None, sum(item["frame_count"] for item in group), True)
            if structured:
                if (scene_contract_issue(text, expected, task, check_content=False)
                        or has_recovered_scene_text(parse_scene_response(text))):
                    return _join_scene_windows(observations, task, "Synthesis was incomplete. The original window descriptions are retained in scene order.")
                value = parse_scene_response(text)
                for key, source in (("start_pose", "scene_start_pose"), ("end_pose", "scene_end_pose")):
                    value[key] = boundary[source]
                text = json.dumps(value, ensure_ascii=False)
            merged.append({"response": text, "shot": group[0]["shot"], "last_shot": group[-1]["last_shot"], "start": group[0]["start"],
                           "end": group[-1]["end"], "frame_count": sum(item["frame_count"] for item in group)})
        if len(merged) == len(current):
            if structured:
                return _join_scene_windows(observations, task, "The window evidence exceeded the synthesis context. The original descriptions are retained in scene order.")
            raise ValueError(
                f"The window observations exceed the synthesis context. No adjacent pair fits the "
                f"{budget:,}-character merge allowance (Context size {context_size:,} tokens, "
                f"Max tokens {max_tokens:,}). Increase Context size or reduce Max tokens "
                "in Generation or the connected settings pipe."
            )
        current = merged
    return current[0]["response"]


def analyze_sequence(frames, *, prompt, task, generate, instance, grid, capacity=8,
                     overlap=2, grid_side=2048, context_size=8192, max_tokens=2048):
    """One output per source, including when it needs several bounded requests."""
    observations, requests = [], []
    sent, omitted = {}, set()
    last_data = {}
    width = height = 0
    measure_camera = SCENE_TASK_MODES.get(task) == "motion" and not _context(prompt)[0].get("excluded_source_cuts")
    with closing(visual_windows(frames, capacity, min(overlap, capacity - 1))) as windows:
        for number, window in enumerate(windows):
            check_interrupt()
            original_count = len(window)
            if task in SCENE_TASK_MODES:
                omitted.update(frame.position for frame in window if not torch.count_nonzero(frame.pixels))
                window = [frame for frame in window if frame.position not in omitted]
                if not window:
                    continue
            batch = torch.stack([frame.pixels for frame in window])
            height, width = batch.shape[1:3]
            request, layout = pack_grid(window, instance, grid_side) if grid else (batch, None)
            scale = measured_shot_scale(window) if measure_camera and len(window) == original_count else None
            geometry = {"frame_scale_evidence": scale} if scale else {}
            request_prompt = _window_prompt(prompt, window, batch, layout, number, scale)
            text, last_data = generate(request_prompt, request, len(window), False)
            text, original_camera = reconcile_camera_response(text, scale)
            observations.append({"response": text, "shot": window[0].shot, "last_shot": window[-1].shot,
                                 "start": window[0].position, "end": window[-1].position,
                                 "frame_count": len(window), **geometry})
            requests.append({"positions": [frame.position for frame in window], "shot": window[0].shot,
                             "seconds": [frame.seconds for frame in window],
                             "image_count": len(request), "width": request.shape[2], "height": request.shape[1],
                             **geometry, **original_camera, **({"grid": layout} if layout else {})})
            sent.update({frame.position: frame.seconds for frame in window})
            del batch, request, window
    if not observations:
        if not omitted:
            raise ValueError("No frames were supplied for visual analysis")
        text = black_scene_observation(prompt, task)
    elif len(observations) == 1:
        text = observations[0]["response"]
    else:
        text = _merge(observations, prompt, task, generate, context_size, max_tokens)
    evidence = {"positions": sorted(sent), "image_count": len(sent), "width": width, "height": height,
                "presentation": "grid" if grid else "frames", "windows": requests,
                "omitted_black_positions": sorted(omitted), "image_handling": "frame_sequence"}
    if (len(observations) > 1 and all("frame_scale_evidence" in item for item in observations)
            and len({(item["shot"], item["frame_scale_evidence"]["framing"]) for item in observations}) == 1):
        text, original = reconcile_camera_response(text, observations[0]["frame_scale_evidence"])
        if original:
            evidence["synthesis_model_camera_wording"] = original["model_camera_wording"]
    if omitted and not sent:
        evidence["skipped_reason"] = "all_black"
    if any(value is not None for value in sent.values()):
        evidence["seconds"] = [sent[index] for index in sorted(sent)]
        evidence["timing"] = "estimated from native VIDEO frame rate"
    return text, last_data, evidence


def attach_sequence_evidence(text, evidence, *, longest_side, every_nth):
    try:
        value = parse_scene_response(text)
    except ValueError:
        return text
    if not isinstance(value, dict):
        return text
    return encode_scene_evidence(value, {**evidence, "longest_side": longest_side, "every_nth": every_nth})
