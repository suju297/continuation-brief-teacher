from __future__ import annotations

import gc
import json
import os
import re
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_TRANSCRIPT_TURN_CHAR_LIMIT = 1200
DEFAULT_TRANSCRIPT_CHAR_BUDGET = 14000
DEFAULT_DEBUG_TEXT_CHAR_LIMIT = 6000

PREAMBLE = "We are continuing an earlier conversation, not starting fresh."
THINK_BLOCK_PATTERN = re.compile(r"<think>([\s\S]*?)</think>", re.IGNORECASE)
REQUIRED_SECTIONS = [
    PREAMBLE,
    "Objective:",
    "Established context:",
    "Decisions already made:",
    "Rejected paths / do not revisit unless necessary:",
    "Open questions:",
    "Where we left off:",
    "Continue from this exact state.",
    "My next request:",
]
SECTION_LINE_PATTERNS = {
    PREAMBLE: re.compile(rf"^{re.escape(PREAMBLE)}$", re.MULTILINE),
    "Objective:": re.compile(r"^Objective:$", re.MULTILINE),
    "Established context:": re.compile(r"^Established context:$", re.MULTILINE),
    "Decisions already made:": re.compile(r"^Decisions already made:$", re.MULTILINE),
    "Rejected paths / do not revisit unless necessary:": re.compile(
        r"^Rejected paths / do not revisit unless necessary:$",
        re.MULTILINE,
    ),
    "Open questions:": re.compile(r"^Open questions:$", re.MULTILINE),
    "Where we left off:": re.compile(r"^Where we left off:$", re.MULTILINE),
    "Continue from this exact state.": re.compile(r"^Continue from this exact state\.$", re.MULTILINE),
    "My next request:": re.compile(r"^My next request:$", re.MULTILINE),
}
PROMPT_ECHO_MARKERS = (
    "\nRequirements:\n",
    "\nKey requirements:\n",
    "Exactly the structure and section order:",
    "Rewrite the supplied conversation state into the exact brief template below.",
    "Return exactly this structure:",
    "\nSource transcript:\n",
    "Do not invent goals, decisions, constraints, files, APIs, or next steps.",
    "We are given a transcript",
    "We have the rolling state helper provided",
    "We have the transcript from the conversation",
    "Let's break down",
)
PLACEHOLDER_BULLET_PATTERN = re.compile(r"(?m)^-\s*\.\.\.\s*$")
INLINE_SECTION_PATTERNS = {
    "Objective:": re.compile(r"(?mi)^\s*(?:\*\*)?\s*Objective\s*(?:\*\*)?\s*:[ \t]+(.+\S)\s*$"),
    "Established context:": re.compile(
        r"(?mi)^\s*(?:\*\*)?\s*Established context\s*(?:\*\*)?\s*:[ \t]+(.+\S)\s*$"
    ),
    "Decisions already made:": re.compile(
        r"(?mi)^\s*(?:\*\*)?\s*Decisions already made\s*(?:\*\*)?\s*:[ \t]+(.+\S)\s*$"
    ),
    "Rejected paths / do not revisit unless necessary:": re.compile(
        r"(?mi)^\s*(?:\*\*)?\s*Rejected paths / do not revisit unless necessary\s*(?:\*\*)?\s*:[ \t]+(.+\S)\s*$"
    ),
    "Open questions:": re.compile(r"(?mi)^\s*(?:\*\*)?\s*Open questions\s*(?:\*\*)?\s*:[ \t]+(.+\S)\s*$"),
    "Where we left off:": re.compile(r"(?mi)^\s*(?:\*\*)?\s*Where we left off\s*(?:\*\*)?\s*:[ \t]+(.+\S)\s*$"),
    "My next request:": re.compile(r"(?mi)^\s*(?:\*\*)?\s*My next request\s*(?:\*\*)?\s*:[ \t]+(.+\S)\s*$"),
}
SECTION_HEADER_NORMALIZATIONS = (
    (re.compile(rf"(?mi)^\s*{re.escape(PREAMBLE)}\s*$"), PREAMBLE),
    (re.compile(r"(?mi)^\s*(?:\*\*)?\s*Objective\s*(?:\*\*)?\s*:\s*$"), "Objective:"),
    (re.compile(r"(?mi)^\s*(?:\*\*)?\s*Established context\s*(?:\*\*)?\s*:\s*$"), "Established context:"),
    (re.compile(r"(?mi)^\s*(?:\*\*)?\s*Decisions already made\s*(?:\*\*)?\s*:\s*$"), "Decisions already made:"),
    (
        re.compile(r"(?mi)^\s*(?:\*\*)?\s*Rejected paths / do not revisit unless necessary\s*(?:\*\*)?\s*:\s*$"),
        "Rejected paths / do not revisit unless necessary:",
    ),
    (re.compile(r"(?mi)^\s*(?:\*\*)?\s*Open questions\s*(?:\*\*)?\s*:\s*$"), "Open questions:"),
    (re.compile(r"(?mi)^\s*(?:\*\*)?\s*Where we left off\s*(?:\*\*)?\s*:\s*$"), "Where we left off:"),
    (re.compile(r"(?mi)^\s*(?:\*\*)?\s*Continue from this exact state\.?\s*$"), "Continue from this exact state."),
    (re.compile(r"(?mi)^\s*(?:\*\*)?\s*My next request\s*(?:\*\*)?\s*:\s*$"), "My next request:"),
)
KNOWN_INPUT_PATHS = {
    "full": "/kaggle/input/qwen4b-teacher-2gpu-inputs/teacher_input_rows.json",
    "repair": "/kaggle/input/qwen4b-teacher-2gpu-repair-inputs/live_extension_review_queue_parallel_assistant_repaired_v11.json",
}


class MalformedBriefError(ValueError):
    def __init__(self, issues: list[str], debug_payload: dict[str, Any]) -> None:
        self.issues = issues
        self.debug_payload = debug_payload
        super().__init__(f"Model returned a malformed continuation brief: {summarize_issues(issues)}")


def ensure_parent(path: str | None) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)


def sanitize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def trim_text(text: Any, max_chars: int) -> str:
    compact = sanitize_text(text)
    if len(compact) <= max_chars:
        return compact
    return f"{compact[:max_chars].strip()}..."


def trim_debug_text(text: Any, max_chars: int = DEFAULT_DEBUG_TEXT_CHAR_LIMIT) -> str:
    value = str(text or "")
    if len(value) <= max_chars:
        return value
    remaining = len(value) - max_chars
    return f"{value[:max_chars].rstrip()}\n...[truncated {remaining} chars]"


def build_transcript_text(
    transcript_turns: list[dict[str, Any]],
    *,
    per_turn_char_limit: int,
    total_char_budget: int,
) -> str:
    rendered_turns = [
        f"{index + 1}. {str(turn.get('role', '')).upper()}: {trim_text(turn.get('text', ''), per_turn_char_limit)}"
        for index, turn in enumerate(transcript_turns)
    ]
    if not rendered_turns:
        return "No transcript turns were provided."
    if total_char_budget <= 0:
        return "\n".join(rendered_turns)

    kept_turns: list[str] = []
    used_chars = 0
    for turn_text in reversed(rendered_turns):
        turn_size = len(turn_text) + 1
        if kept_turns and used_chars + turn_size > total_char_budget:
            break
        kept_turns.append(turn_text)
        used_chars += turn_size
        if used_chars >= total_char_budget:
            break

    kept_turns.reverse()
    omitted_turns = len(rendered_turns) - len(kept_turns)
    if omitted_turns > 0:
        kept_turns.insert(0, f"[{omitted_turns} earlier turns omitted to fit the transcript budget.]")
    return "\n".join(kept_turns)


def render_list(title: str, items: list[str], fallback: str) -> str:
    lines = items if items else [fallback]
    return f"{title}:\n" + "\n".join(f"- {item}" for item in lines)


def render_template_list(items: list[str], fallback: str) -> str:
    lines = [item for item in items if item]
    return "\n".join(f"- {item}" for item in (lines or [fallback]))


def build_prompt(
    row: dict[str, Any],
    capture_rationale: bool,
    *,
    transcript_char_budget: int = DEFAULT_TRANSCRIPT_CHAR_BUDGET,
    per_turn_char_limit: int = DEFAULT_TRANSCRIPT_TURN_CHAR_LIMIT,
) -> str:
    rolling_state = row.get("rolling_state") or {}
    transcript_turns = row.get("transcript_turns") or []
    transcript_text = build_transcript_text(
        transcript_turns,
        per_turn_char_limit=per_turn_char_limit,
        total_char_budget=transcript_char_budget,
    )
    objective = render_template_list(
        [sanitize_text(rolling_state.get("objective"))] if sanitize_text(rolling_state.get("objective")) else [],
        "Objective still needs confirmation.",
    )
    context_lines = render_template_list(
        [sanitize_text(item) for item in (rolling_state.get("constraints") or []) if sanitize_text(item)],
        "No explicit constraints have been captured yet.",
    )
    decision_lines = render_template_list(
        [sanitize_text(item) for item in (rolling_state.get("decisions") or []) if sanitize_text(item)],
        "No firm decisions are recorded yet.",
    )
    rejected_lines = render_template_list(
        [sanitize_text(item) for item in (rolling_state.get("rejected") or []) if sanitize_text(item)],
        "No rejected paths have been recorded yet.",
    )
    open_question_lines = render_template_list(
        [sanitize_text(item) for item in (rolling_state.get("open_questions") or []) if sanitize_text(item)],
        "No open questions are active right now.",
    )
    next_step_lines = render_template_list(
        [sanitize_text(rolling_state.get("next_step"))] if sanitize_text(rolling_state.get("next_step")) else [],
        "Resume from the latest visible turn.",
    )

    instructions = [
        "Rewrite the supplied conversation state into the exact brief template below.",
        "Keep every section header exactly as written.",
        "Do not add any extra headers, titles, dates, explanations, or code fences.",
        "Use the transcript as the source of truth and the rolling state only as a helper.",
        "If something is unclear or not explicit in the transcript, leave it out.",
        "If the supplied content is already concise and faithful, copy it with only minimal edits.",
        "Return only the final brief text.",
    ]
    if capture_rationale:
        instructions.extend(
            [
                "If you show visible reasoning, place it inside <think>...</think>.",
                "After the closing </think> tag, return only the final continuation brief.",
            ]
        )

    return "\n".join(
        [
            *instructions,
            "",
            "Source transcript:",
            f"- title: {row.get('title', '')}",
            f"- provider: {row.get('provider', '')}",
            transcript_text or "No transcript turns were provided.",
            "",
            "Rolling state helper:",
            render_list(
                "Objective",
                [sanitize_text(rolling_state.get("objective"))] if sanitize_text(rolling_state.get("objective")) else [],
                "Objective still needs confirmation.",
            ),
            "",
            render_list(
                "Established context",
                [sanitize_text(item) for item in (rolling_state.get("constraints") or []) if sanitize_text(item)],
                "No explicit constraints have been captured yet.",
            ),
            "",
            render_list(
                "Decisions already made",
                [sanitize_text(item) for item in (rolling_state.get("decisions") or []) if sanitize_text(item)],
                "No firm decisions are recorded yet.",
            ),
            "",
            render_list(
                "Rejected paths / do not revisit unless necessary",
                [sanitize_text(item) for item in (rolling_state.get("rejected") or []) if sanitize_text(item)],
                "No rejected paths have been recorded yet.",
            ),
            "",
            render_list(
                "Open questions",
                [sanitize_text(item) for item in (rolling_state.get("open_questions") or []) if sanitize_text(item)],
                "No open questions are active right now.",
            ),
            "",
            render_list(
                "Where we left off",
                [sanitize_text(rolling_state.get("next_step"))] if sanitize_text(rolling_state.get("next_step")) else [],
                "Resume from the latest visible turn.",
            ),
            "",
            "Return exactly this structure:",
            "",
            PREAMBLE,
            "",
            "Objective:",
            objective,
            "",
            "Established context:",
            context_lines,
            "",
            "Decisions already made:",
            decision_lines,
            "",
            "Rejected paths / do not revisit unless necessary:",
            rejected_lines,
            "",
            "Open questions:",
            open_question_lines,
            "",
            "Where we left off:",
            next_step_lines,
            "",
            "Continue from this exact state.",
            "My next request:",
            "- ",
        ]
    )


def sanitize_model_brief(text: Any) -> str:
    value = str(text or "")
    value = re.sub(r"^```[a-z]*\n?", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\n```$", "", value, flags=re.IGNORECASE)
    return value.strip()


def extract_tagged_rationale(text: str) -> tuple[str, str]:
    chunks: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        cleaned = sanitize_model_brief(match.group(1))
        if cleaned:
            chunks.append(cleaned)
        return ""

    remaining = THINK_BLOCK_PATTERN.sub(_replace, text)
    return "\n\n".join(chunks).strip(), remaining.strip()


def split_model_output(raw_text: str) -> tuple[str, str]:
    sanitized_raw = sanitize_model_brief(raw_text).replace("\r", "")
    auxiliary_rationale, remaining = extract_tagged_rationale(sanitized_raw)
    return remaining or sanitized_raw, auxiliary_rationale


def normalize_generated_brief(text: str) -> str:
    brief = sanitize_model_brief(text).replace("\r", "")
    brief = re.sub(r"\n{3,}", "\n\n", brief)
    brief = re.sub(r"(?m)^-\s*", "- ", brief)
    brief = re.sub(
        rf"^(?:{re.escape(PREAMBLE)}\s*\n+){{2,}}",
        f"{PREAMBLE}\n\n",
        brief,
        count=1,
    )
    for section, pattern in INLINE_SECTION_PATTERNS.items():
        brief = pattern.sub(lambda match, section=section: f"{section}\n- {match.group(1).strip()}", brief)
    for pattern, replacement in SECTION_HEADER_NORMALIZATIONS:
        brief = pattern.sub(replacement, brief)
    brief = re.sub(r"(?m)^My next request:\s*$\n(?!-)", "My next request:\n-", brief)
    brief = re.sub(r"(?m)^My next request:\n-\s*$", "My next request:\n-", brief)
    return brief.strip()


def has_required_sections(text: str) -> bool:
    return all(section in text for section in REQUIRED_SECTIONS)


def has_duplicate_or_missing_section_lines(text: str) -> bool:
    return any(len(pattern.findall(text)) != 1 for pattern in SECTION_LINE_PATTERNS.values())


def has_prompt_echo(text: str) -> bool:
    return any(marker in text[:1200] for marker in PROMPT_ECHO_MARKERS)


def has_single_next_request_bullet(text: str) -> bool:
    match = re.search(r"(?ms)^My next request:\n(.*)\Z", text)
    if not match:
        return False
    lines = [line.strip() for line in match.group(1).splitlines() if line.strip()]
    return len(lines) == 1 and (lines[0] == "-" or lines[0].startswith("- "))


def describe_invalid_brief(text: str) -> list[str]:
    issues: list[str] = []
    missing_sections = [section for section in REQUIRED_SECTIONS if section not in text]
    if missing_sections:
        preview = ", ".join(missing_sections[:4])
        if len(missing_sections) > 4:
            preview = f"{preview}, +{len(missing_sections) - 4} more"
        issues.append(f"missing sections: {preview}")

    for section, pattern in SECTION_LINE_PATTERNS.items():
        count = len(pattern.findall(text))
        if count == 0:
            issues.append(f"missing section line: {section}")
        elif count > 1:
            issues.append(f"duplicate section line: {section} x{count}")

    if not has_single_next_request_bullet(text):
        issues.append("invalid My next request bullet")
    if PLACEHOLDER_BULLET_PATTERN.search(text):
        issues.append("contains placeholder bullet")
    if has_prompt_echo(text):
        issues.append("prompt echo in opening text")
    if not issues:
        issues.append("unknown validation failure")
    return issues


def summarize_issues(issues: list[str], limit: int = 4) -> str:
    if not issues:
        return "unknown validation failure"
    summary = "; ".join(issues[:limit])
    if len(issues) > limit:
        summary = f"{summary}; +{len(issues) - limit} more"
    return summary


def is_valid_generated_brief(text: str) -> bool:
    return (
        has_required_sections(text)
        and not has_duplicate_or_missing_section_lines(text)
        and has_single_next_request_bullet(text)
        and not PLACEHOLDER_BULLET_PATTERN.search(text)
        and not has_prompt_echo(text)
    )


def rewrite_tags(tags: list[str], model: str, has_rationale: bool) -> list[str]:
    filtered = [
        tag
        for tag in list(tags or [])
        if tag
        not in {
            "teacher-draft-backfilled",
            "teacher-draft-saved",
            "teacher-draft-generated",
            "teacher-draft-failed",
            "teacher-rationale-captured",
            "local-heuristic",
        }
        and not str(tag).startswith("local-model:")
    ]
    filtered.extend(["teacher-draft-generated", f"local-model:{model}"])
    if has_rationale:
        filtered.append("teacher-rationale-captured")
    unique: list[str] = []
    for tag in filtered:
        if tag not in unique:
            unique.append(tag)
    return unique


def rewrite_failure_tags(tags: list[str], model: str) -> list[str]:
    filtered = [
        tag
        for tag in list(tags or [])
        if tag
        not in {
            "teacher-draft-backfilled",
            "teacher-draft-saved",
            "teacher-draft-generated",
            "teacher-draft-failed",
            "teacher-rationale-captured",
            "local-heuristic",
        }
        and not str(tag).startswith("local-model:")
    ]
    filtered.extend(["teacher-draft-failed", f"local-model:{model}"])
    unique: list[str] = []
    for tag in filtered:
        if tag not in unique:
            unique.append(tag)
    return unique


def finalize_failed_row(row: dict[str, Any], model: str) -> dict[str, Any]:
    next_row = dict(row)
    next_row.pop("teacher_draft_brief", None)
    next_row.pop("auxiliary_rationale", None)
    next_row.pop("auxiliary_rationale_format", None)
    next_row["tags"] = rewrite_failure_tags(list(row.get("tags", [])), model)
    return next_row


def count_rows_with_tag(rows: list[dict[str, Any]], tag: str) -> int:
    return sum(1 for row in rows if tag in list(row.get("tags", [])))


def build_generation_attempts(requested_max_new_tokens: int) -> list[dict[str, int]]:
    raw_budgets = [
        DEFAULT_TRANSCRIPT_CHAR_BUDGET,
        min(DEFAULT_TRANSCRIPT_CHAR_BUDGET, 10000),
        min(DEFAULT_TRANSCRIPT_CHAR_BUDGET, 7000),
        min(DEFAULT_TRANSCRIPT_CHAR_BUDGET, 5000),
    ]
    raw_token_caps = [
        requested_max_new_tokens,
        min(requested_max_new_tokens, 320),
        min(requested_max_new_tokens, 256),
        min(requested_max_new_tokens, 192),
    ]
    attempts: list[dict[str, int]] = []
    for transcript_char_budget, max_new_tokens in zip(raw_budgets, raw_token_caps):
        attempt = {
            "transcript_char_budget": transcript_char_budget,
            "max_new_tokens": max_new_tokens,
        }
        if attempt not in attempts:
            attempts.append(attempt)
    return attempts


def is_cuda_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message and ("cuda" in message or "cublas" in message)


def clear_torch_memory(device: torch.device | None) -> None:
    gc.collect()
    if not device or device.type != "cuda" or not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except RuntimeError:
        pass
    try:
        torch.cuda.ipc_collect()
    except RuntimeError:
        pass


def detect_single_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("This notebook requires a CUDA GPU.")
    return torch.device("cuda")


def resolve_model_load_settings(device_index: int = 0) -> dict[str, Any]:
    major, minor = torch.cuda.get_device_capability(device_index)
    capability = (major, minor)
    if capability < (7, 0):
        return {
            "capability": capability,
            "torch_dtype": torch.float16,
            "attn_implementation": "eager",
            "compatibility_mode": "legacy_cuda",
        }
    return {
        "capability": capability,
        "torch_dtype": torch.float16,
        "attn_implementation": "sdpa",
        "compatibility_mode": "default",
    }


def list_input_json_candidates(root: str = "/kaggle/input") -> list[str]:
    input_root = Path(root)
    if not input_root.exists():
        return []
    return [str(path) for path in sorted(input_root.rglob("*.json"))]


def resolve_explicit_input_path(explicit_input_path: str) -> str:
    candidate = Path(explicit_input_path)
    if candidate.is_file():
        return str(candidate)
    if candidate.is_dir():
        json_candidates = sorted(candidate.rglob("*.json"))
        if len(json_candidates) == 1:
            return str(json_candidates[0])
        for preferred_name in (
            "teacher_input_rows.json",
            "live_extension_review_queue_parallel_assistant_repaired_v11.json",
        ):
            for json_candidate in json_candidates:
                if json_candidate.name == preferred_name:
                    return str(json_candidate)
        preview = "\n".join(str(path) for path in json_candidates) if json_candidates else "(no JSON files found)"
        raise FileNotFoundError(
            f"Explicit input directory does not resolve to a single JSON file: {candidate}\n"
            f"Candidates:\n{preview}"
        )
    raise FileNotFoundError(f"Explicit input path does not exist: {candidate}")


def resolve_known_input_path(run_dataset: str) -> str:
    dataset_key = run_dataset.strip().lower()
    if dataset_key not in KNOWN_INPUT_PATHS:
        raise ValueError(f"Unknown RUN_DATASET value: {run_dataset}")
    candidate = Path(KNOWN_INPUT_PATHS[dataset_key])
    if candidate.exists():
        return str(candidate)
    available = list_input_json_candidates()
    preview = "\n".join(available) if available else "(no JSON files found under /kaggle/input)"
    raise FileNotFoundError(
        f"Expected input JSON is missing for RUN_DATASET={run_dataset}: {candidate}\n"
        f"Available candidates:\n{preview}"
    )


def resolve_input_path(run_dataset: str, explicit_input_path: str = "") -> str:
    explicit_value = explicit_input_path.strip()
    if explicit_value:
        return resolve_explicit_input_path(explicit_value)
    return resolve_known_input_path(run_dataset)


def load_json_rows(path: str) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError(f"{path} does not contain a JSON array.")
    return data


def select_rows(rows: list[dict[str, Any]], limit: int = 0) -> list[dict[str, Any]]:
    return rows[:limit] if limit > 0 else list(rows)


def derive_failure_debug_path(workdir: str, run_mode: str) -> str:
    return str(Path(workdir) / f"{run_mode}.failures.ndjson")


def append_failure_debug_record(debug_path: str, payload: dict[str, Any]) -> None:
    ensure_parent(debug_path)
    with Path(debug_path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def reset_run_artifacts(*paths: str) -> None:
    for path in paths:
        if path and Path(path).exists():
            Path(path).unlink()


def load_resume_rows(output_path: str, expected_rows: int) -> list[dict[str, Any]]:
    path = Path(output_path)
    if not path.exists():
        return []
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, list):
        raise RuntimeError("Existing output file does not contain a JSON array.")
    if len(loaded) > expected_rows:
        raise RuntimeError("Existing output file has more rows than the selected run.")
    return loaded


def validate_resume_summary(
    summary_output_path: str | None,
    *,
    input_path: str,
    model: str,
    run_mode: str,
    requested_rows: int,
) -> None:
    if not summary_output_path:
        return
    summary_path = Path(summary_output_path)
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checks = {
        "inputPath": input_path,
        "model": model,
        "runMode": run_mode,
        "requestedRows": requested_rows,
    }
    mismatches = [
        f"{key}={summary.get(key)!r} expected {value!r}"
        for key, value in checks.items()
        if summary.get(key) != value
    ]
    if mismatches:
        raise RuntimeError("Resume refused because prior summary does not match this run: " + "; ".join(mismatches))


def build_summary(
    *,
    model: str,
    input_path: str,
    run_mode: str,
    requested_rows: int,
    completed_rows: int,
    generated_rows: list[dict[str, Any]],
    failures: list[dict[str, str]],
    capture_rationale: bool,
    started_at: float,
    resumed_rows: int,
    status: str,
    failure_debug_path: str,
) -> dict[str, Any]:
    success_count = count_rows_with_tag(generated_rows, "teacher-draft-generated")
    failure_count = count_rows_with_tag(generated_rows, "teacher-draft-failed")
    shard_complete = completed_rows == requested_rows
    return {
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "durationMs": round((time.time() - started_at) * 1000),
        "model": model,
        "inputPath": input_path,
        "runMode": run_mode,
        "inputRows": requested_rows,
        "requestedRows": requested_rows,
        "completedRows": completed_rows,
        "mergedRows": completed_rows,
        "generatedRows": success_count,
        "captureRationale": capture_rationale,
        "rationaleCapturedRows": sum(1 for row in generated_rows if str(row.get("auxiliary_rationale", "")).strip()),
        "failedRows": failure_count,
        "failures": failures,
        "numShards": 1,
        "completeShards": 1 if shard_complete else 0,
        "shards": [
            {
                "shardIndex": 0,
                "path": "single-run",
                "expectedRows": requested_rows,
                "mergedRows": completed_rows,
                "complete": shard_complete,
            }
        ],
        "shardIndex": 0,
        "resumedRows": resumed_rows,
        "status": status,
        "failureDebugPath": failure_debug_path,
    }


def write_checkpoint(
    *,
    output_path: str,
    summary_output: str | None,
    generated_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    ensure_parent(output_path)
    Path(output_path).write_text(json.dumps(generated_rows, indent=2) + "\n", encoding="utf-8")
    if summary_output:
        ensure_parent(summary_output)
        Path(summary_output).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def read_failure_records(debug_path: str, limit: int = 5) -> list[dict[str, Any]]:
    path = Path(debug_path)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(records) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


class HFTeacherGenerator:
    def __init__(self, model_id: str, temperature: float, max_new_tokens: int) -> None:
        self.model_id = model_id
        self.device = detect_single_device()
        self.model_load_settings = resolve_model_load_settings(0)
        self.cuda_capability = self.model_load_settings["capability"]
        self.attn_implementation = self.model_load_settings["attn_implementation"]
        self.torch_dtype = self.model_load_settings["torch_dtype"]
        self.compatibility_mode = self.model_load_settings["compatibility_mode"]
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": self.torch_dtype,
            "attn_implementation": self.attn_implementation,
        }
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        except TypeError:
            model_kwargs.pop("attn_implementation", None)
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt: str, capture_rationale: bool, *, max_new_tokens: int | None = None) -> str:
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict[str, Any] = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            if "qwen" in self.model_id.lower() and not capture_rationale:
                kwargs["enable_thinking"] = False
            try:
                inputs = self.tokenizer.apply_chat_template(messages, **kwargs)
            except TypeError:
                kwargs.pop("enable_thinking", None)
                inputs = self.tokenizer.apply_chat_template(messages, **kwargs)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt")

        inputs = inputs.to(self.device)
        generated = None
        new_tokens = None
        try:
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    do_sample=self.temperature > 0,
                    temperature=max(self.temperature, 1e-5),
                    max_new_tokens=max_new_tokens or self.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            new_tokens = generated[:, inputs["input_ids"].shape[1] :]
            return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
        finally:
            del inputs
            if new_tokens is not None:
                del new_tokens
            if generated is not None:
                del generated
            clear_torch_memory(self.device)


def generate_teacher_draft(
    *,
    generator: HFTeacherGenerator,
    row: dict[str, Any],
    capture_rationale: bool,
) -> tuple[str, str]:
    attempts = build_generation_attempts(generator.max_new_tokens)
    last_oom_error: RuntimeError | None = None
    attempt_debug_records: list[dict[str, Any]] = []

    for attempt_index, attempt in enumerate(attempts, start=1):
        prompt = build_prompt(
            row,
            capture_rationale,
            transcript_char_budget=attempt["transcript_char_budget"],
        )
        try:
            raw_output = generator.generate(
                prompt,
                capture_rationale,
                max_new_tokens=attempt["max_new_tokens"],
            )
            brief_text, auxiliary_rationale = split_model_output(raw_output)
            brief = normalize_generated_brief(brief_text)
            initial_brief = brief
            initial_issues = describe_invalid_brief(brief) if not is_valid_generated_brief(brief) else []

            if initial_issues:
                print(
                    f"Invalid brief for '{row.get('title', '')}' on attempt {attempt_index}/{len(attempts)}: "
                    f"{summarize_issues(initial_issues)}",
                    flush=True,
                )
                retry_prompt = (
                    f"{prompt}\n\nThe previous answer was invalid because it repeated the instructions, "
                    "used placeholders, or missed the required structure. Regenerate and output only the "
                    "final continuation brief with filled sections and no analysis."
                )
                retry_raw_output = generator.generate(
                    retry_prompt,
                    capture_rationale,
                    max_new_tokens=attempt["max_new_tokens"],
                )
                brief_text, auxiliary_rationale = split_model_output(retry_raw_output)
                brief = normalize_generated_brief(brief_text)
                retry_issues = describe_invalid_brief(brief) if not is_valid_generated_brief(brief) else []

                if retry_issues:
                    print(
                        f"Retry still invalid for '{row.get('title', '')}' on attempt {attempt_index}/{len(attempts)}: "
                        f"{summarize_issues(retry_issues)}",
                        flush=True,
                    )
                    attempt_debug_records.append(
                        {
                            "attemptIndex": attempt_index,
                            "transcriptCharBudget": attempt["transcript_char_budget"],
                            "maxNewTokens": attempt["max_new_tokens"],
                            "promptLength": len(prompt),
                            "promptPreview": trim_debug_text(prompt),
                            "initialRawOutput": trim_debug_text(raw_output),
                            "initialNormalizedBrief": trim_debug_text(initial_brief),
                            "initialIssues": initial_issues,
                            "retryRawOutput": trim_debug_text(retry_raw_output),
                            "retryNormalizedBrief": trim_debug_text(brief),
                            "retryIssues": retry_issues,
                        }
                    )
                    continue

            return brief, auxiliary_rationale
        except RuntimeError as exc:
            if not is_cuda_oom_error(exc):
                raise
            last_oom_error = exc
            print(
                "OOM while generating "
                f"'{row.get('title', '')}' on attempt {attempt_index}/{len(attempts)}; "
                f"retrying with transcript_char_budget={attempt['transcript_char_budget']} "
                f"and max_new_tokens={attempt['max_new_tokens']}",
                flush=True,
            )
            clear_torch_memory(generator.device)

    if attempt_debug_records:
        issues: list[str] = []
        for record in attempt_debug_records:
            for issue in record.get("retryIssues") or record.get("initialIssues") or []:
                if issue not in issues:
                    issues.append(issue)
        raise MalformedBriefError(issues, {"attempts": attempt_debug_records})

    if last_oom_error is not None:
        raise RuntimeError(f"{last_oom_error} after {len(attempts)} memory-recovery attempts.") from last_oom_error

    raise RuntimeError("Teacher draft generation exhausted all attempts without producing output.")


def run_generation(
    *,
    generator: HFTeacherGenerator,
    rows: list[dict[str, Any]],
    input_path: str,
    output_path: str,
    summary_output: str | None,
    workdir: str,
    model: str,
    run_mode: str,
    checkpoint_every: int,
    capture_rationale: bool,
    resume: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started_at = time.time()
    selected_rows = list(rows)
    failure_debug_path = derive_failure_debug_path(workdir, run_mode)

    if resume:
        validate_resume_summary(
            summary_output,
            input_path=input_path,
            model=model,
            run_mode=run_mode,
            requested_rows=len(selected_rows),
        )
        generated_rows = load_resume_rows(output_path, len(selected_rows))
        if generated_rows:
            print(f"Resuming {run_mode} run from row {len(generated_rows) + 1}/{len(selected_rows)}", flush=True)
    else:
        reset_run_artifacts(output_path, summary_output or "", failure_debug_path)
        generated_rows = []

    failures: list[dict[str, str]] = []
    resumed_rows = len(generated_rows)

    for index, row in enumerate(selected_rows[resumed_rows:], start=resumed_rows + 1):
        print(f"Generating {index}/{len(selected_rows)}: {row.get('title', '')}", flush=True)
        try:
            brief, auxiliary_rationale = generate_teacher_draft(
                generator=generator,
                row=row,
                capture_rationale=capture_rationale,
            )
            next_row = dict(row)
            next_row["teacher_draft_brief"] = brief
            if auxiliary_rationale:
                next_row["auxiliary_rationale"] = auxiliary_rationale
                next_row["auxiliary_rationale_format"] = "visible_cot"
            next_row["tags"] = rewrite_tags(list(row.get("tags", [])), model, bool(auxiliary_rationale))
            generated_rows.append(next_row)
        except Exception as exc:  # noqa: BLE001
            failure_record = {
                "title": str(row.get("title", "")),
                "conversation_id": str(row.get("conversation_id", "")),
                "message": str(exc),
            }
            debug_payload: dict[str, Any] = {
                "title": failure_record["title"],
                "conversation_id": failure_record["conversation_id"],
                "provider": str(row.get("provider", "")),
                "model": model,
                "runMode": run_mode,
                "errorType": type(exc).__name__,
                "message": str(exc),
            }
            if isinstance(exc, MalformedBriefError):
                debug_payload.update(exc.debug_payload)
            append_failure_debug_record(failure_debug_path, debug_payload)
            print(
                f"Logged failure debug for '{failure_record['title']}' to {failure_debug_path}",
                flush=True,
            )
            failures.append(failure_record)
            generated_rows.append(finalize_failed_row(row, model))

        if len(generated_rows) % max(checkpoint_every, 1) == 0 or index == len(selected_rows):
            summary = build_summary(
                model=model,
                input_path=input_path,
                run_mode=run_mode,
                requested_rows=len(selected_rows),
                completed_rows=len(generated_rows),
                generated_rows=generated_rows,
                failures=failures,
                capture_rationale=capture_rationale,
                started_at=started_at,
                resumed_rows=resumed_rows,
                status="running" if index < len(selected_rows) else "completed",
                failure_debug_path=failure_debug_path,
            )
            write_checkpoint(
                output_path=output_path,
                summary_output=summary_output,
                generated_rows=generated_rows,
                summary=summary,
            )

    final_summary = build_summary(
        model=model,
        input_path=input_path,
        run_mode=run_mode,
        requested_rows=len(selected_rows),
        completed_rows=len(generated_rows),
        generated_rows=generated_rows,
        failures=failures,
        capture_rationale=capture_rationale,
        started_at=started_at,
        resumed_rows=resumed_rows,
        status="completed",
        failure_debug_path=failure_debug_path,
    )
    write_checkpoint(
        output_path=output_path,
        summary_output=summary_output,
        generated_rows=generated_rows,
        summary=final_summary,
    )
    return generated_rows, final_summary
