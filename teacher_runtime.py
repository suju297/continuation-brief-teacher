from __future__ import annotations

import gc
import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean, median
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_STAGE1_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_STAGE2_MODELS = (
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B-Instruct-2507",
)
PROMPT_METHODS = (
    "baseline",
    "strict_template",
    "heuristic_rewrite",
)
DEFAULT_PROMPT_METHOD = "strict_template"
DEFAULT_TRANSCRIPT_TURN_CHAR_LIMIT = 1200
DEFAULT_TRANSCRIPT_CHAR_BUDGET = 14000
DEFAULT_DEBUG_TEXT_CHAR_LIMIT = 6000
DEFAULT_SMOKE_LIMIT = 10

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
CONTENT_SECTION_HEADERS = REQUIRED_SECTIONS[1:]
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
STATUS_PRIORITY = {
    "hard_reject": 0,
    "soft_accept": 1,
    "format_clean": 2,
}
GENERIC_FALLBACK_LINES = {
    "Objective still needs confirmation.",
    "No explicit constraints have been captured yet.",
    "No firm decisions are recorded yet.",
    "No rejected paths have been recorded yet.",
    "No open questions are active right now.",
    "Resume from the latest visible turn.",
}
SECTION_KEYS = (
    "objective",
    "constraints",
    "decisions",
    "rejected",
    "open_questions",
    "next_step",
)
SECTION_WEIGHTS = {
    "objective": 0.20,
    "constraints": 0.15,
    "decisions": 0.20,
    "rejected": 0.10,
    "open_questions": 0.10,
    "next_step": 0.25,
}
ACTIONABLE_REINTRO_SECTIONS = ("objective", "decisions", "next_step")
STATE_MATCH_THRESHOLD = 0.58
SECTION_LABELS = {
    "objective": "Objective:",
    "constraints": "Established context:",
    "decisions": "Decisions already made:",
    "rejected": "Rejected paths / do not revisit unless necessary:",
    "open_questions": "Open questions:",
    "next_step": "Where we left off:",
}
SCALAR_METRIC_NAMES = (
    "state_fidelity_score",
    "teacher_similarity_score",
    "must_include_recall",
    "avoid_hit_rate",
    "rejected_path_reintroduction_rate",
    "carry_forward_accuracy",
    "unwarranted_change_rate",
    "next_step_fidelity",
    "brevity_readability_score",
    "hallucination_penalty",
)


@dataclass
class BriefSections:
    objective: str
    constraints: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    next_step: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BriefSections":
        return cls(
            objective=str(raw.get("objective", "")).strip(),
            constraints=[str(item).strip() for item in raw.get("constraints", []) if str(item).strip()],
            decisions=[str(item).strip() for item in raw.get("decisions", []) if str(item).strip()],
            rejected=[str(item).strip() for item in raw.get("rejected", []) if str(item).strip()],
            open_questions=[
                str(item).strip()
                for item in raw.get("open_questions", raw.get("openQuestions", []))
                if str(item).strip()
            ],
            next_step=str(raw.get("next_step", raw.get("nextStep", ""))).strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def section_items(self) -> dict[str, list[str]]:
        return {
            "objective": [self.objective] if self.objective else [],
            "constraints": list(self.constraints),
            "decisions": list(self.decisions),
            "rejected": list(self.rejected),
            "open_questions": list(self.open_questions),
            "next_step": [self.next_step] if self.next_step else [],
        }


@dataclass
class ExperimentMetrics:
    state_fidelity_score: float
    teacher_similarity_score: float
    must_include_recall: float
    avoid_hit_rate: float
    rejected_path_reintroduction_rate: float
    carry_forward_accuracy: float
    unwarranted_change_rate: float
    next_step_fidelity: float
    brevity_readability_score: float
    hallucination_penalty: float
    section_scores: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationOutcome:
    brief: str
    auxiliary_rationale: str
    status: str
    issues: list[str]
    debug_payload: dict[str, Any]
    duration_ms: int


def ensure_parent(path: str | None) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_json(path: str | Path, payload: Any) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return str(path)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def slugify(value: str) -> str:
    lowered = value.lower().strip()
    lowered = re.sub(r"[^a-z0-9]+", "-", lowered)
    return lowered.strip("-") or "value"


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return mean(values)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def tokenize(value: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9']+", normalize_text(value)) if token]


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "we",
    "with",
    "you",
}


def content_tokens(value: str) -> list[str]:
    return [token for token in tokenize(value) if token not in STOP_WORDS]


def unique_content_tokens(value: str) -> set[str]:
    return set(content_tokens(value))


def jaccard_similarity(left: str, right: str) -> float:
    left_tokens = unique_content_tokens(left)
    right_tokens = unique_content_tokens(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def sequence_similarity(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio()


def text_similarity(left: str, right: str) -> float:
    return 0.6 * jaccard_similarity(left, right) + 0.4 * sequence_similarity(left, right)


def novel_token_ratio(output: str, evidence: str) -> float:
    output_tokens = unique_content_tokens(output)
    evidence_tokens = unique_content_tokens(evidence)
    if not output_tokens:
        return 0.0
    return len(output_tokens - evidence_tokens) / len(output_tokens)


def _clean_item(value: str) -> str:
    cleaned = value.strip()
    if cleaned.startswith("-"):
        cleaned = cleaned[1:].strip()
    return cleaned


def parse_continuation_brief(text: str) -> BriefSections:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    section_items: dict[str, list[str]] = {key: [] for key in SECTION_KEYS}
    seen_headers: set[str] = set()
    active_section: str | None = None

    header_map = {
        "Objective:": "objective",
        "Established context:": "constraints",
        "Decisions already made:": "decisions",
        "Rejected paths / do not revisit unless necessary:": "rejected",
        "Open questions:": "open_questions",
        "Where we left off:": "next_step",
    }

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        header_key = header_map.get(line)
        if header_key:
            active_section = header_key
            seen_headers.add(header_key)
            continue

        if line.startswith((PREAMBLE, "Continue from this exact state.", "My next request:")):
            active_section = None
            continue

        if active_section is None:
            continue

        item = _clean_item(line)
        if not item:
            continue
        section_items[active_section].append(item)

    missing = [header for header in header_map.values() if header not in seen_headers]
    if missing:
        raise ValueError(f"Continuation brief is missing required sections: {', '.join(missing)}")

    if not section_items["objective"]:
        raise ValueError("Continuation brief did not contain an objective item.")
    if not section_items["next_step"]:
        raise ValueError("Continuation brief did not contain a next_step item.")

    return BriefSections(
        objective=section_items["objective"][0],
        constraints=section_items["constraints"],
        decisions=section_items["decisions"],
        rejected=section_items["rejected"],
        open_questions=section_items["open_questions"],
        next_step=section_items["next_step"][0],
    )


def _list_f1(predicted: list[str], gold: list[str]) -> float:
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0

    recall = mean(max(text_similarity(gold_item, predicted_item) for predicted_item in predicted) for gold_item in gold)
    precision = mean(max(text_similarity(predicted_item, gold_item) for gold_item in gold) for predicted_item in predicted)
    if recall + precision == 0:
        return 0.0
    return 2 * recall * precision / (recall + precision)


def score_sections(predicted: BriefSections | None, gold: BriefSections) -> dict[str, float]:
    if predicted is None:
        return {section: 0.0 for section in SECTION_KEYS}

    predicted_items = predicted.section_items()
    gold_items = gold.section_items()
    return {section: _list_f1(predicted_items[section], gold_items[section]) for section in SECTION_KEYS}


def _state_item_matches(left: str, right: str) -> bool:
    return text_similarity(left, right) >= STATE_MATCH_THRESHOLD


def _carry_forward_metrics(row: dict[str, Any], predicted: BriefSections | None) -> tuple[float, float]:
    prior_items = BriefSections.from_dict(row.get("rolling_state") or {}).section_items()
    gold_items = BriefSections.from_dict(row.get("gold_sections") or {}).section_items()
    predicted_items = predicted.section_items() if predicted is not None else {section: [] for section in SECTION_KEYS}

    persistent_targets: list[tuple[str, str, str]] = []
    for section in SECTION_KEYS:
        for prior_item in prior_items[section]:
            matching_gold = next(
                (gold_item for gold_item in gold_items[section] if _state_item_matches(prior_item, gold_item)),
                None,
            )
            if matching_gold:
                persistent_targets.append((section, prior_item, matching_gold))

    if not persistent_targets:
        return 1.0, 0.0

    persisted = 0
    unwarranted_changes = 0
    for section, prior_item, gold_item in persistent_targets:
        target_items = predicted_items[section]
        if any(_state_item_matches(prior_item, item) or _state_item_matches(gold_item, item) for item in target_items):
            persisted += 1
        else:
            unwarranted_changes += 1

    total = len(persistent_targets)
    return persisted / total, unwarranted_changes / total


def _rejected_path_reintroduction_rate(
    predicted: BriefSections | None,
    generated_brief: str,
    expected_avoid: list[str],
    rejected_items: list[str],
) -> float:
    if predicted is None:
        actionable_items = [generated_brief]
    else:
        predicted_items = predicted.section_items()
        actionable_items = [
            item
            for section in ACTIONABLE_REINTRO_SECTIONS
            for item in predicted_items[section]
        ]

    actionable_text = "\n".join(actionable_items).lower()
    if expected_avoid:
        hits = sum(1 for term in expected_avoid if term.lower() in actionable_text)
        return hits / len(expected_avoid)

    if not rejected_items:
        return 0.0

    hits = 0
    for rejected_item in rejected_items:
        if any(_state_item_matches(rejected_item, candidate) for candidate in actionable_items):
            hits += 1
    return hits / len(rejected_items)


def score_output(row: dict[str, Any], generated_brief: str, predicted_sections: BriefSections | None) -> ExperimentMetrics:
    gold_sections = BriefSections.from_dict(row.get("gold_sections") or {})
    section_scores = score_sections(predicted_sections, gold_sections)
    state_fidelity = sum(section_scores[section] * SECTION_WEIGHTS[section] for section in SECTION_KEYS)

    output_lower = generated_brief.lower()
    expected_must_include = [str(item).strip() for item in row.get("expected_must_include", []) if str(item).strip()]
    expected_avoid = [str(item).strip() for item in row.get("expected_avoid", []) if str(item).strip()]
    must_hits = sum(1 for term in expected_must_include if term.lower() in output_lower)
    must_recall = 1.0 if not expected_must_include else must_hits / len(expected_must_include)

    carry_forward_accuracy, unwarranted_change_rate = _carry_forward_metrics(row, predicted_sections)
    rejected_path_reintroduction_rate = _rejected_path_reintroduction_rate(
        predicted_sections,
        generated_brief,
        expected_avoid,
        gold_sections.rejected,
    )
    avoid_hit_rate = rejected_path_reintroduction_rate

    required_headers = sum(1 for label in SECTION_LABELS.values() if label in generated_brief)
    structure_score = required_headers / len(SECTION_LABELS)
    length_ok = 1.0 if 250 <= len(generated_brief) <= 2200 else 0.0
    brevity_readability = 0.5 * structure_score + 0.5 * length_ok

    teacher_similarity = sequence_similarity(generated_brief, str(row.get("teacher_draft_brief", "")))
    evidence = "\n".join(str(turn.get("text", "")) for turn in row.get("transcript_turns", [])) + "\n" + str(
        row.get("teacher_draft_brief", "")
    )
    hallucination_penalty = max(0.0, novel_token_ratio(generated_brief, evidence) - 0.38)

    return ExperimentMetrics(
        state_fidelity_score=round(state_fidelity, 4),
        teacher_similarity_score=round(teacher_similarity, 4),
        must_include_recall=round(must_recall, 4),
        avoid_hit_rate=round(avoid_hit_rate, 4),
        rejected_path_reintroduction_rate=round(rejected_path_reintroduction_rate, 4),
        carry_forward_accuracy=round(carry_forward_accuracy, 4),
        unwarranted_change_rate=round(unwarranted_change_rate, 4),
        next_step_fidelity=round(section_scores["next_step"], 4),
        brevity_readability_score=round(brevity_readability, 4),
        hallucination_penalty=round(hallucination_penalty, 4),
        section_scores={section: round(score, 4) for section, score in section_scores.items()},
    )


def aggregate_gold_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_dicts: list[ExperimentMetrics] = []
    for row in rows:
        if not str(row.get("gold_brief", "")).strip():
            continue
        generated_brief = str(row.get("teacher_draft_brief", "")).strip()
        predicted_sections = None
        if generated_brief:
            try:
                predicted_sections = parse_continuation_brief(generated_brief)
            except Exception:
                predicted_sections = None
        metric_dicts.append(score_output(row, generated_brief, predicted_sections))

    if not metric_dicts:
        return {"count": 0}

    payload: dict[str, Any] = {"count": len(metric_dicts)}
    for metric_name in SCALAR_METRIC_NAMES:
        payload[metric_name] = round(safe_mean([getattr(metrics, metric_name) for metrics in metric_dicts]), 4)
    return payload


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


def compose_heuristic_brief(state: dict[str, Any]) -> str:
    return "\n".join(
        [
            PREAMBLE,
            "",
            render_list(
                "Objective",
                [sanitize_text(state.get("objective"))] if sanitize_text(state.get("objective")) else [],
                "Objective still needs confirmation.",
            ),
            "",
            render_list(
                "Established context",
                [sanitize_text(item) for item in (state.get("constraints") or []) if sanitize_text(item)],
                "No explicit constraints have been captured yet.",
            ),
            "",
            render_list(
                "Decisions already made",
                [sanitize_text(item) for item in (state.get("decisions") or []) if sanitize_text(item)],
                "No firm decisions are recorded yet.",
            ),
            "",
            render_list(
                "Rejected paths / do not revisit unless necessary",
                [sanitize_text(item) for item in (state.get("rejected") or []) if sanitize_text(item)],
                "No rejected paths have been recorded yet.",
            ),
            "",
            render_list(
                "Open questions",
                [sanitize_text(item) for item in (state.get("open_questions", state.get("openQuestions", [])) or []) if sanitize_text(item)],
                "No open questions are active right now.",
            ),
            "",
            render_list(
                "Where we left off",
                [sanitize_text(state.get("next_step", state.get("nextStep")))] if sanitize_text(state.get("next_step", state.get("nextStep"))) else [],
                "Resume from the latest visible turn.",
            ),
            "",
            "Continue from this exact state.",
            "My next request:",
            "- ",
        ]
    )


def build_recent_turns(row: dict[str, Any]) -> str:
    turns = [f"- {turn.get('role', '')}: {turn.get('text', '')}" for turn in row.get("transcript_turns", [])]
    return "\n".join(turns) if turns else "- No recent turns provided."


def build_baseline_prompt(
    row: dict[str, Any],
    capture_rationale: bool,
    *,
    transcript_char_budget: int,
    per_turn_char_limit: int,
) -> str:
    del capture_rationale
    state = row.get("rolling_state") or {}
    transcript_text = build_transcript_text(
        row.get("transcript_turns") or [],
        per_turn_char_limit=per_turn_char_limit,
        total_char_budget=transcript_char_budget,
    )
    return "\n".join(
        [
            PREAMBLE,
            "",
            f"Chat title: {row.get('title', '')}",
            f"Provider: {row.get('provider', '')}",
            "",
            render_list(
                "Objective",
                [sanitize_text(state.get("objective"))] if sanitize_text(state.get("objective")) else [],
                "Objective still needs confirmation.",
            ),
            "",
            render_list(
                "Established context",
                [sanitize_text(item) for item in (state.get("constraints") or []) if sanitize_text(item)],
                "No explicit constraints.",
            ),
            "",
            render_list(
                "Decisions already made",
                [sanitize_text(item) for item in (state.get("decisions") or []) if sanitize_text(item)],
                "No fixed decisions.",
            ),
            "",
            render_list(
                "Rejected paths / do not revisit unless necessary",
                [sanitize_text(item) for item in (state.get("rejected") or []) if sanitize_text(item)],
                "No rejected paths.",
            ),
            "",
            render_list(
                "Open questions",
                [sanitize_text(item) for item in (state.get("open_questions") or []) if sanitize_text(item)],
                "No open questions.",
            ),
            "",
            render_list(
                "Where we left off",
                [sanitize_text(state.get("next_step"))] if sanitize_text(state.get("next_step")) else [],
                "Resume from latest visible turn.",
            ),
            "",
            "Recent turns:",
            transcript_text if transcript_text else build_recent_turns(row),
            "",
            "Write the final continuation brief now.",
        ]
    )


def build_strict_template_prompt(
    row: dict[str, Any],
    capture_rationale: bool,
    *,
    transcript_char_budget: int,
    per_turn_char_limit: int,
) -> str:
    rolling_state = row.get("rolling_state") or {}
    transcript_text = build_transcript_text(
        row.get("transcript_turns") or [],
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


def build_heuristic_rewrite_prompt(
    row: dict[str, Any],
    capture_rationale: bool,
    *,
    transcript_char_budget: int,
    per_turn_char_limit: int,
) -> str:
    del capture_rationale
    draft = compose_heuristic_brief(row.get("rolling_state") or {})
    transcript_text = build_transcript_text(
        row.get("transcript_turns") or [],
        per_turn_char_limit=per_turn_char_limit,
        total_char_budget=transcript_char_budget,
    )
    return "\n".join(
        [
            "You are editing a continuation brief draft.",
            "Preserve the exact section headers and order.",
            "Keep facts faithful to the supplied state.",
            "If the draft is already good, return it unchanged.",
            "",
            "Recent turns for style only:",
            transcript_text if transcript_text else build_recent_turns(row),
            "",
            "Draft to revise:",
            draft,
        ]
    )


def build_prompt(
    row: dict[str, Any],
    capture_rationale: bool,
    *,
    prompt_method: str = DEFAULT_PROMPT_METHOD,
    transcript_char_budget: int = DEFAULT_TRANSCRIPT_CHAR_BUDGET,
    per_turn_char_limit: int = DEFAULT_TRANSCRIPT_TURN_CHAR_LIMIT,
) -> str:
    if prompt_method == "baseline":
        return build_baseline_prompt(
            row,
            capture_rationale,
            transcript_char_budget=transcript_char_budget,
            per_turn_char_limit=per_turn_char_limit,
        )
    if prompt_method == "heuristic_rewrite":
        return build_heuristic_rewrite_prompt(
            row,
            capture_rationale,
            transcript_char_budget=transcript_char_budget,
            per_turn_char_limit=per_turn_char_limit,
        )
    if prompt_method != "strict_template":
        raise ValueError(f"Unknown prompt method: {prompt_method}")
    return build_strict_template_prompt(
        row,
        capture_rationale,
        transcript_char_budget=transcript_char_budget,
        per_turn_char_limit=per_turn_char_limit,
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


def count_present_content_headers(text: str) -> int:
    return sum(1 for section in CONTENT_SECTION_HEADERS if section in text)


def is_near_empty_brief(text: str) -> bool:
    return len(re.sub(r"[^A-Za-z0-9]+", "", text)) < 40


def has_both_tail_anchors_missing(text: str) -> bool:
    return "Where we left off:" not in text and "My next request:" not in text


def looks_like_unusable_scaffolding(text: str) -> bool:
    meaningful_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line in REQUIRED_SECTIONS or line == "-":
            continue
        cleaned = _clean_item(line)
        if not cleaned or cleaned in GENERIC_FALLBACK_LINES:
            continue
        meaningful_lines.append(cleaned)
    return len(meaningful_lines) < 2


def describe_format_issues(text: str) -> list[str]:
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
    return issues


def summarize_issues(issues: list[str], limit: int = 4) -> str:
    if not issues:
        return "no issues"
    summary = "; ".join(issues[:limit])
    if len(issues) > limit:
        summary = f"{summary}; +{len(issues) - limit} more"
    return summary


def is_format_clean(text: str) -> bool:
    return (
        has_required_sections(text)
        and not has_duplicate_or_missing_section_lines(text)
        and has_single_next_request_bullet(text)
        and not PLACEHOLDER_BULLET_PATTERN.search(text)
        and not has_prompt_echo(text)
    )


def classify_generated_brief(text: str) -> tuple[str, list[str]]:
    hard_reject_issues: list[str] = []
    if is_near_empty_brief(text):
        hard_reject_issues.append("empty or near-empty normalized output")
    if has_prompt_echo(text):
        hard_reject_issues.append("prompt echo in opening text")
    if PLACEHOLDER_BULLET_PATTERN.search(text):
        hard_reject_issues.append("contains placeholder bullet")
    if count_present_content_headers(text) < 4:
        hard_reject_issues.append("section coverage below 4/8 required content headers")
    if has_both_tail_anchors_missing(text):
        hard_reject_issues.append("missing both continuation tail anchors")
    if looks_like_unusable_scaffolding(text):
        hard_reject_issues.append("obvious unusable scaffolding")
    if hard_reject_issues:
        return "hard_reject", hard_reject_issues
    if is_format_clean(text):
        return "format_clean", []
    format_issues = describe_format_issues(text)
    return "soft_accept", format_issues or ["format drift"]


def rewrite_tags(tags: list[str], model: str, status: str, has_rationale: bool) -> list[str]:
    filtered = [
        tag
        for tag in list(tags or [])
        if tag
        not in {
            "teacher-draft-backfilled",
            "teacher-draft-saved",
            "teacher-draft-generated",
            "teacher-draft-failed",
            "teacher-draft-format-clean",
            "teacher-draft-soft-accepted",
            "teacher-rationale-captured",
            "local-heuristic",
        }
        and not str(tag).startswith("local-model:")
    ]
    if status == "hard_reject":
        filtered.extend(["teacher-draft-failed", f"local-model:{model}"])
    else:
        filtered.extend(["teacher-draft-generated", f"local-model:{model}"])
        if status == "format_clean":
            filtered.append("teacher-draft-format-clean")
        elif status == "soft_accept":
            filtered.append("teacher-draft-soft-accepted")
    if has_rationale and status != "hard_reject":
        filtered.append("teacher-rationale-captured")
    unique: list[str] = []
    for tag in filtered:
        if tag not in unique:
            unique.append(tag)
    return unique


def apply_generation_metadata(
    row: dict[str, Any],
    *,
    status: str,
    issues: list[str],
    model: str,
    prompt_method: str,
    duration_ms: int,
    auxiliary_rationale: str = "",
    brief: str = "",
) -> dict[str, Any]:
    next_row = dict(row)
    next_row["teacher_generation_status"] = status
    next_row["teacher_validation_issues"] = list(issues)
    next_row["teacher_generation_model"] = model
    next_row["teacher_generation_prompt"] = prompt_method
    next_row["teacher_generation_duration_ms"] = duration_ms
    if status == "hard_reject":
        next_row.pop("teacher_draft_brief", None)
        next_row.pop("auxiliary_rationale", None)
        next_row.pop("auxiliary_rationale_format", None)
    else:
        next_row["teacher_draft_brief"] = brief
        if auxiliary_rationale:
            next_row["auxiliary_rationale"] = auxiliary_rationale
            next_row["auxiliary_rationale_format"] = "visible_cot"
        else:
            next_row.pop("auxiliary_rationale", None)
            next_row.pop("auxiliary_rationale_format", None)
    next_row["tags"] = rewrite_tags(list(row.get("tags", [])), model, status, bool(auxiliary_rationale))
    return next_row


def count_rows_with_status(rows: list[dict[str, Any]], status: str) -> int:
    return sum(1 for row in rows if row.get("teacher_generation_status") == status)


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
            f"Explicit input directory does not resolve to a single JSON file: {candidate}\nCandidates:\n{preview}"
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
        f"Expected input JSON is missing for RUN_DATASET={run_dataset}: {candidate}\nAvailable candidates:\n{preview}"
    )


def resolve_input_path(run_dataset: str, explicit_input_path: str = "") -> str:
    explicit_value = explicit_input_path.strip()
    if explicit_value:
        return resolve_explicit_input_path(explicit_value)
    return resolve_known_input_path(run_dataset)


def load_json_rows(path: str) -> list[dict[str, Any]]:
    data = read_json(path)
    if not isinstance(data, list):
        raise RuntimeError(f"{path} does not contain a JSON array.")
    return data


def select_rows(rows: list[dict[str, Any]], limit: int = 0) -> list[dict[str, Any]]:
    return rows[:limit] if limit > 0 else list(rows)


def select_split_rows(rows: list[dict[str, Any]], split: str, *, require_gold: bool = False) -> list[dict[str, Any]]:
    selected = [row for row in rows if str(row.get("split", "")).strip() == split]
    if require_gold:
        selected = [row for row in selected if str(row.get("gold_brief", "")).strip()]
    return selected


def transcript_length_bucket(row: dict[str, Any]) -> str:
    turn_count = len(row.get("transcript_turns") or [])
    if turn_count <= 3:
        return "short"
    if turn_count <= 9:
        return "medium"
    return "long"


def _sorted_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (str(row.get("conversation_id", "")), str(row.get("case_id", ""))))


def build_stage1_calibration_slice(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    train_gold_rows = _sorted_rows(select_split_rows(rows, "train", require_gold=True))
    bucket_targets = {"short": 4, "medium": 4, "long": 4}
    selected_by_bucket: dict[str, list[dict[str, Any]]] = {}

    for bucket_name, target_count in bucket_targets.items():
        bucket_rows = [row for row in train_gold_rows if transcript_length_bucket(row) == bucket_name]
        if len(bucket_rows) < target_count:
            raise RuntimeError(f"Not enough gold-backed train rows for bucket={bucket_name}: need {target_count}, found {len(bucket_rows)}")
        selected_by_bucket[bucket_name] = bucket_rows[:target_count]

    selected_case_ids = {str(row.get("case_id", "")) for bucket_rows in selected_by_bucket.values() for row in bucket_rows}
    selected_claude_count = sum(
        1 for bucket_rows in selected_by_bucket.values() for row in bucket_rows if str(row.get("provider", "")) == "claude"
    )
    if selected_claude_count < 2:
        claude_candidates = [
            row
            for row in train_gold_rows
            if str(row.get("provider", "")) == "claude" and str(row.get("case_id", "")) not in selected_case_ids
        ]
        for candidate in claude_candidates:
            bucket_name = transcript_length_bucket(candidate)
            replacement_pool = [row for row in reversed(selected_by_bucket[bucket_name]) if str(row.get("provider", "")) != "claude"]
            if not replacement_pool:
                continue
            replacement = replacement_pool[0]
            replace_index = selected_by_bucket[bucket_name].index(replacement)
            selected_by_bucket[bucket_name][replace_index] = candidate
            selected_case_ids.remove(str(replacement.get("case_id", "")))
            selected_case_ids.add(str(candidate.get("case_id", "")))
            selected_claude_count += 1
            if selected_claude_count >= 2:
                break

    selected_rows: list[dict[str, Any]] = []
    for bucket_name in ("short", "medium", "long"):
        selected_rows.extend(_sorted_rows(selected_by_bucket[bucket_name]))
    return selected_rows


def derive_failure_debug_path(workdir: str, run_mode: str) -> str:
    return str(Path(workdir) / f"{run_mode}.failures.ndjson")


def derive_soft_accept_review_path(workdir: str, run_mode: str) -> str:
    return str(Path(workdir) / f"{run_mode}.soft_accept_review.json")


def derive_stage_summary_path(workdir: str, stage_name: str) -> str:
    return str(Path(workdir) / stage_name / "stage_summary.json")


def arm_paths(base_workdir: str, stage_name: str, arm_name: str) -> dict[str, str]:
    arm_dir = Path(base_workdir) / stage_name / slugify(arm_name)
    return {
        "workdir": str(arm_dir),
        "output": str(arm_dir / "qwen4b_teacher_drafts_merged.json"),
        "summary": str(arm_dir / "qwen4b_teacher_drafts_merged.summary.json"),
    }


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
    prompt_method: str,
    run_mode: str,
    requested_rows: int,
) -> None:
    if not summary_output_path:
        return
    summary_path = Path(summary_output_path)
    if not summary_path.exists():
        return
    summary = read_json(summary_path)
    checks = {
        "inputPath": input_path,
        "model": model,
        "promptMethod": prompt_method,
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


def build_soft_accept_review_entries(rows: list[dict[str, Any]], limit: int = 0) -> list[dict[str, Any]]:
    soft_rows = [row for row in rows if row.get("teacher_generation_status") == "soft_accept"]
    entries = [
        {
            "case_id": row.get("case_id", ""),
            "conversation_id": row.get("conversation_id", ""),
            "title": row.get("title", ""),
            "provider": row.get("provider", ""),
            "split": row.get("split", ""),
            "transcriptTurnCount": len(row.get("transcript_turns") or []),
            "validationIssues": list(row.get("teacher_validation_issues", [])),
            "teacherDraftBrief": row.get("teacher_draft_brief", ""),
        }
        for row in soft_rows
    ]
    if limit > 0:
        return entries[:limit]
    return entries


def write_soft_accept_review_queue(review_path: str, rows: list[dict[str, Any]]) -> str:
    return write_json(review_path, build_soft_accept_review_entries(rows))


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


def read_review_queue(review_path: str, limit: int = 5) -> list[dict[str, Any]]:
    path = Path(review_path)
    if not path.exists():
        return []
    records = read_json(path)
    if limit <= 0:
        return list(records)
    return list(records)[:limit]


def build_summary(
    *,
    model: str,
    prompt_method: str,
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
    soft_accept_review_path: str,
) -> dict[str, Any]:
    format_clean_rows = count_rows_with_status(generated_rows, "format_clean")
    soft_accept_rows = count_rows_with_status(generated_rows, "soft_accept")
    hard_reject_rows = count_rows_with_status(generated_rows, "hard_reject")
    usable_rows = format_clean_rows + soft_accept_rows
    reason_counts: Counter[str] = Counter()
    durations: list[int] = []
    for row in generated_rows:
        for issue in row.get("teacher_validation_issues", []):
            if row.get("teacher_generation_status") == "hard_reject":
                reason_counts[str(issue)] += 1
        duration_value = row.get("teacher_generation_duration_ms")
        if isinstance(duration_value, (int, float)):
            durations.append(int(duration_value))

    gold_metrics = aggregate_gold_metrics(generated_rows)
    shard_complete = completed_rows == requested_rows
    return {
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "durationMs": round((time.time() - started_at) * 1000),
        "medianRuntimePerRowMs": int(median(durations)) if durations else 0,
        "model": model,
        "promptMethod": prompt_method,
        "inputPath": input_path,
        "runMode": run_mode,
        "inputRows": requested_rows,
        "requestedRows": requested_rows,
        "completedRows": completed_rows,
        "mergedRows": completed_rows,
        "generatedRows": usable_rows,
        "formatCleanRows": format_clean_rows,
        "softAcceptedRows": soft_accept_rows,
        "hardRejectRows": hard_reject_rows,
        "failedRows": hard_reject_rows,
        "usableYield": round(safe_ratio(usable_rows, requested_rows), 4),
        "hardRejectRate": round(safe_ratio(hard_reject_rows, requested_rows), 4),
        "promptEchoHardRejects": reason_counts.get("prompt echo in opening text", 0),
        "hardRejectReasonCounts": dict(sorted(reason_counts.items())),
        "captureRationale": capture_rationale,
        "rationaleCapturedRows": sum(1 for row in generated_rows if str(row.get("auxiliary_rationale", "")).strip()),
        "failures": failures,
        "goldBackedMetrics": gold_metrics,
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
        "softAcceptReviewPath": soft_accept_review_path,
    }


def write_checkpoint(
    *,
    output_path: str,
    summary_output: str | None,
    generated_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    write_json(output_path, generated_rows)
    if summary_output:
        write_json(summary_output, summary)
    write_soft_accept_review_queue(summary["softAcceptReviewPath"], generated_rows)


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
            "dtype": self.torch_dtype,
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

    def close(self) -> None:
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        clear_torch_memory(self.device)


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[int, int, int]:
    return (
        STATUS_PRIORITY[candidate["status"]],
        -len(candidate["issues"]),
        len(candidate["brief"]),
    )


def _select_better_candidate(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any] | None:
    if left is None:
        return right
    if right is None:
        return left
    return max((left, right), key=_candidate_sort_key)


def _evaluate_generation_candidate(
    *,
    raw_output: str,
    prompt: str,
    attempt_index: int,
    transcript_char_budget: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    brief_text, auxiliary_rationale = split_model_output(raw_output)
    brief = normalize_generated_brief(brief_text)
    status, issues = classify_generated_brief(brief)
    return {
        "status": status,
        "issues": issues,
        "brief": brief,
        "auxiliary_rationale": auxiliary_rationale,
        "raw_output": raw_output,
        "prompt": prompt,
        "attempt_index": attempt_index,
        "transcript_char_budget": transcript_char_budget,
        "max_new_tokens": max_new_tokens,
    }


def generate_teacher_draft(
    *,
    generator: HFTeacherGenerator,
    row: dict[str, Any],
    capture_rationale: bool,
    prompt_method: str,
) -> GenerationOutcome:
    attempts = build_generation_attempts(generator.max_new_tokens)
    last_oom_error: RuntimeError | None = None
    attempt_debug_records: list[dict[str, Any]] = []
    best_candidate: dict[str, Any] | None = None
    started_at = time.time()

    for attempt_index, attempt in enumerate(attempts, start=1):
        prompt = build_prompt(
            row,
            capture_rationale,
            prompt_method=prompt_method,
            transcript_char_budget=attempt["transcript_char_budget"],
        )
        try:
            raw_output = generator.generate(
                prompt,
                capture_rationale,
                max_new_tokens=attempt["max_new_tokens"],
            )
            initial_candidate = _evaluate_generation_candidate(
                raw_output=raw_output,
                prompt=prompt,
                attempt_index=attempt_index,
                transcript_char_budget=attempt["transcript_char_budget"],
                max_new_tokens=attempt["max_new_tokens"],
            )

            chosen_candidate = initial_candidate
            retry_candidate: dict[str, Any] | None = None
            if initial_candidate["status"] != "format_clean":
                print(
                    f"{initial_candidate['status']} for '{row.get('title', '')}' on attempt {attempt_index}/{len(attempts)}: "
                    f"{summarize_issues(initial_candidate['issues'])}",
                    flush=True,
                )
                retry_prompt = (
                    f"{prompt}\n\nThe previous answer had structural or content problems. "
                    "Regenerate the continuation brief with the same facts, no commentary, no template echo, "
                    "and a filled My next request section."
                )
                retry_raw_output = generator.generate(
                    retry_prompt,
                    capture_rationale,
                    max_new_tokens=attempt["max_new_tokens"],
                )
                retry_candidate = _evaluate_generation_candidate(
                    raw_output=retry_raw_output,
                    prompt=retry_prompt,
                    attempt_index=attempt_index,
                    transcript_char_budget=attempt["transcript_char_budget"],
                    max_new_tokens=attempt["max_new_tokens"],
                )
                chosen_candidate = _select_better_candidate(initial_candidate, retry_candidate) or initial_candidate

            best_candidate = _select_better_candidate(best_candidate, chosen_candidate)
            attempt_debug_records.append(
                {
                    "attemptIndex": attempt_index,
                    "transcriptCharBudget": attempt["transcript_char_budget"],
                    "maxNewTokens": attempt["max_new_tokens"],
                    "promptLength": len(prompt),
                    "promptPreview": trim_debug_text(prompt),
                    "initialRawOutput": trim_debug_text(raw_output),
                    "initialNormalizedBrief": trim_debug_text(initial_candidate["brief"]),
                    "initialStatus": initial_candidate["status"],
                    "initialIssues": initial_candidate["issues"],
                    "retryRawOutput": trim_debug_text(retry_candidate["raw_output"]) if retry_candidate else "",
                    "retryNormalizedBrief": trim_debug_text(retry_candidate["brief"]) if retry_candidate else "",
                    "retryStatus": retry_candidate["status"] if retry_candidate else "",
                    "retryIssues": retry_candidate["issues"] if retry_candidate else [],
                    "chosenStatus": chosen_candidate["status"],
                    "chosenIssues": chosen_candidate["issues"],
                }
            )

            if chosen_candidate["status"] == "format_clean":
                break
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

    if best_candidate is not None:
        duration_ms = round((time.time() - started_at) * 1000)
        return GenerationOutcome(
            brief=best_candidate["brief"],
            auxiliary_rationale=best_candidate["auxiliary_rationale"],
            status=best_candidate["status"],
            issues=list(best_candidate["issues"]),
            debug_payload={
                "attempts": attempt_debug_records,
                "chosenStatus": best_candidate["status"],
                "chosenIssues": list(best_candidate["issues"]),
            },
            duration_ms=duration_ms,
        )

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
    prompt_method: str,
    run_mode: str,
    checkpoint_every: int,
    capture_rationale: bool,
    resume: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started_at = time.time()
    selected_rows = list(rows)
    failure_debug_path = derive_failure_debug_path(workdir, run_mode)
    soft_accept_review_path = derive_soft_accept_review_path(workdir, run_mode)

    if resume:
        validate_resume_summary(
            summary_output,
            input_path=input_path,
            model=model,
            prompt_method=prompt_method,
            run_mode=run_mode,
            requested_rows=len(selected_rows),
        )
        generated_rows = load_resume_rows(output_path, len(selected_rows))
        if generated_rows:
            print(f"Resuming {run_mode} run from row {len(generated_rows) + 1}/{len(selected_rows)}", flush=True)
    else:
        reset_run_artifacts(output_path, summary_output or "", failure_debug_path, soft_accept_review_path)
        generated_rows = []

    failures: list[dict[str, str]] = []
    resumed_rows = len(generated_rows)

    for index, row in enumerate(selected_rows[resumed_rows:], start=resumed_rows + 1):
        print(
            f"Generating {index}/{len(selected_rows)} [{prompt_method}] [{model}]: {row.get('title', '')}",
            flush=True,
        )
        try:
            outcome = generate_teacher_draft(
                generator=generator,
                row=row,
                capture_rationale=capture_rationale,
                prompt_method=prompt_method,
            )
            next_row = apply_generation_metadata(
                row,
                status=outcome.status,
                issues=outcome.issues,
                model=model,
                prompt_method=prompt_method,
                duration_ms=outcome.duration_ms,
                auxiliary_rationale=outcome.auxiliary_rationale,
                brief=outcome.brief,
            )
            generated_rows.append(next_row)
            if outcome.status == "hard_reject":
                failure_record = {
                    "title": str(row.get("title", "")),
                    "conversation_id": str(row.get("conversation_id", "")),
                    "message": summarize_issues(outcome.issues),
                }
                failures.append(failure_record)
                append_failure_debug_record(
                    failure_debug_path,
                    {
                        "title": failure_record["title"],
                        "conversation_id": failure_record["conversation_id"],
                        "provider": str(row.get("provider", "")),
                        "model": model,
                        "promptMethod": prompt_method,
                        "runMode": run_mode,
                        "errorType": "HardReject",
                        "message": summarize_issues(outcome.issues),
                        **outcome.debug_payload,
                    },
                )
        except Exception as exc:  # noqa: BLE001
            issues = [f"runtime/model exception: {type(exc).__name__}", str(exc)]
            failure_record = {
                "title": str(row.get("title", "")),
                "conversation_id": str(row.get("conversation_id", "")),
                "message": str(exc),
            }
            failures.append(failure_record)
            generated_rows.append(
                apply_generation_metadata(
                    row,
                    status="hard_reject",
                    issues=issues,
                    model=model,
                    prompt_method=prompt_method,
                    duration_ms=0,
                )
            )
            append_failure_debug_record(
                failure_debug_path,
                {
                    "title": failure_record["title"],
                    "conversation_id": failure_record["conversation_id"],
                    "provider": str(row.get("provider", "")),
                    "model": model,
                    "promptMethod": prompt_method,
                    "runMode": run_mode,
                    "errorType": type(exc).__name__,
                    "message": str(exc),
                },
            )

        if len(generated_rows) % max(checkpoint_every, 1) == 0 or index == len(selected_rows):
            summary = build_summary(
                model=model,
                prompt_method=prompt_method,
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
                soft_accept_review_path=soft_accept_review_path,
            )
            write_checkpoint(
                output_path=output_path,
                summary_output=summary_output,
                generated_rows=generated_rows,
                summary=summary,
            )

    final_summary = build_summary(
        model=model,
        prompt_method=prompt_method,
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
        soft_accept_review_path=soft_accept_review_path,
    )
    write_checkpoint(
        output_path=output_path,
        summary_output=summary_output,
        generated_rows=generated_rows,
        summary=final_summary,
    )
    return generated_rows, final_summary


def _arm_ranking_tuple(summary: dict[str, Any]) -> tuple[float, float, float]:
    metrics = summary.get("goldBackedMetrics", {})
    return (
        float(metrics.get("state_fidelity_score", 0.0)),
        float(metrics.get("next_step_fidelity", 0.0)),
        -float(metrics.get("rejected_path_reintroduction_rate", 1.0)),
    )


def _model_ranking_tuple(summary: dict[str, Any]) -> tuple[float, float, float, float]:
    metrics = summary.get("goldBackedMetrics", {})
    return (
        float(metrics.get("state_fidelity_score", 0.0)),
        float(summary.get("usableYield", 0.0)),
        -float(metrics.get("hallucination_penalty", 1.0)),
        -float(summary.get("medianRuntimePerRowMs", 1e12)),
    )


def stage1_arm_qualifies(summary: dict[str, Any]) -> bool:
    return (
        float(summary.get("usableYield", 0.0)) >= 0.80
        and float(summary.get("hardRejectRate", 1.0)) <= 0.20
        and int(summary.get("promptEchoHardRejects", 1)) == 0
    )


def stage2_arm_qualifies(summary: dict[str, Any]) -> bool:
    metrics = summary.get("goldBackedMetrics", {})
    return (
        float(summary.get("usableYield", 0.0)) >= 0.85
        and float(summary.get("hardRejectRate", 1.0)) <= 0.10
        and float(metrics.get("next_step_fidelity", 0.0)) >= 0.65
        and float(metrics.get("rejected_path_reintroduction_rate", 1.0)) <= 0.10
    )


def stage3_holdout_passes(summary: dict[str, Any], *, validation_state_fidelity: float) -> bool:
    metrics = summary.get("goldBackedMetrics", {})
    holdout_state_fidelity = float(metrics.get("state_fidelity_score", 0.0))
    return (
        float(summary.get("usableYield", 0.0)) >= 0.80
        and float(summary.get("hardRejectRate", 1.0)) <= 0.15
        and abs(holdout_state_fidelity - validation_state_fidelity) <= 0.05
    )


def _sample_rows_by_bucket(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0 or not rows:
        return []
    ordered_rows = _sorted_rows(rows)
    bucket_rows = {
        "short": [row for row in ordered_rows if transcript_length_bucket(row) == "short"],
        "medium": [row for row in ordered_rows if transcript_length_bucket(row) == "medium"],
        "long": [row for row in ordered_rows if transcript_length_bucket(row) == "long"],
    }
    sample: list[dict[str, Any]] = []
    while len(sample) < count:
        progress = False
        for bucket_name in ("short", "medium", "long"):
            if bucket_rows[bucket_name]:
                sample.append(bucket_rows[bucket_name].pop(0))
                progress = True
                if len(sample) >= count:
                    break
        if not progress:
            break
    return sample


def build_review_sample(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    sampled = _sample_rows_by_bucket([row for row in rows if row.get("teacher_generation_status") == "soft_accept"], count)
    return [
        {
            "case_id": row.get("case_id", ""),
            "title": row.get("title", ""),
            "provider": row.get("provider", ""),
            "split": row.get("split", ""),
            "status": row.get("teacher_generation_status", ""),
            "validationIssues": list(row.get("teacher_validation_issues", [])),
        }
        for row in sampled
    ]


def _run_arm(
    *,
    rows: list[dict[str, Any]],
    input_path: str,
    base_workdir: str,
    stage_name: str,
    arm_name: str,
    model: str,
    prompt_method: str,
    checkpoint_every: int,
    capture_rationale: bool,
    temperature: float,
    max_new_tokens: int,
) -> dict[str, Any]:
    paths = arm_paths(base_workdir, stage_name, arm_name)
    generator = HFTeacherGenerator(model, temperature, max_new_tokens)
    try:
        generated_rows, summary = run_generation(
            generator=generator,
            rows=rows,
            input_path=input_path,
            output_path=paths["output"],
            summary_output=paths["summary"],
            workdir=paths["workdir"],
            model=model,
            prompt_method=prompt_method,
            run_mode=stage_name,
            checkpoint_every=checkpoint_every,
            capture_rationale=capture_rationale,
            resume=False,
        )
    finally:
        generator.close()
    return {
        "armName": arm_name,
        "model": model,
        "promptMethod": prompt_method,
        "outputPath": paths["output"],
        "summaryPath": paths["summary"],
        "summary": summary,
        "reviewSample": build_review_sample(generated_rows, 6 if stage_name == "stage2_model" else 10),
    }


def run_stage1_prompt_screen(
    *,
    rows: list[dict[str, Any]],
    input_path: str,
    base_workdir: str,
    checkpoint_every: int,
    capture_rationale: bool,
    temperature: float,
    max_new_tokens: int,
    model: str = DEFAULT_STAGE1_MODEL,
    prompt_methods: tuple[str, ...] = PROMPT_METHODS,
) -> dict[str, Any]:
    calibration_rows = build_stage1_calibration_slice(rows)
    arm_results = [
        _run_arm(
            rows=calibration_rows,
            input_path=input_path,
            base_workdir=base_workdir,
            stage_name="stage1_prompt",
            arm_name=prompt_method,
            model=model,
            prompt_method=prompt_method,
            checkpoint_every=checkpoint_every,
            capture_rationale=capture_rationale,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        for prompt_method in prompt_methods
    ]
    shortlisted = [result for result in arm_results if stage1_arm_qualifies(result["summary"])]
    winner = max(shortlisted, key=lambda result: _arm_ranking_tuple(result["summary"])) if shortlisted else None
    stage_summary = {
        "stage": "stage1_prompt",
        "calibrationRows": len(calibration_rows),
        "selectedCaseIds": [row.get("case_id", "") for row in calibration_rows],
        "selectedTitles": [row.get("title", "") for row in calibration_rows],
        "shortlistedPromptMethods": [result["promptMethod"] for result in shortlisted],
        "winnerPromptMethod": winner["promptMethod"] if winner else None,
        "pass": winner is not None,
        "results": arm_results,
    }
    write_json(derive_stage_summary_path(base_workdir, "stage1_prompt"), stage_summary)
    return stage_summary


def run_stage2_model_selection(
    *,
    rows: list[dict[str, Any]],
    input_path: str,
    base_workdir: str,
    checkpoint_every: int,
    capture_rationale: bool,
    temperature: float,
    max_new_tokens: int,
    prompt_method: str,
    models: tuple[str, ...] = DEFAULT_STAGE2_MODELS,
) -> dict[str, Any]:
    val_rows = select_split_rows(rows, "val")
    arm_results = [
        _run_arm(
            rows=val_rows,
            input_path=input_path,
            base_workdir=base_workdir,
            stage_name="stage2_model",
            arm_name=model,
            model=model,
            prompt_method=prompt_method,
            checkpoint_every=checkpoint_every,
            capture_rationale=capture_rationale,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        for model in models
    ]
    qualified = [result for result in arm_results if stage2_arm_qualifies(result["summary"])]
    ranked_results = sorted(arm_results, key=lambda result: _model_ranking_tuple(result["summary"]), reverse=True)
    winner = max(qualified, key=lambda result: _model_ranking_tuple(result["summary"])) if qualified else None
    runner_up = next((result for result in ranked_results if winner is not None and result["model"] != winner["model"]), None)
    stage_summary = {
        "stage": "stage2_model",
        "validationRows": len(val_rows),
        "goldBackedValidationRows": sum(1 for row in val_rows if str(row.get("gold_brief", "")).strip()),
        "promptMethod": prompt_method,
        "qualifiedModels": [result["model"] for result in qualified],
        "winnerModel": winner["model"] if winner else None,
        "runnerUpModel": runner_up["model"] if runner_up else None,
        "pass": winner is not None,
        "results": arm_results,
    }
    write_json(derive_stage_summary_path(base_workdir, "stage2_model"), stage_summary)
    return stage_summary


def run_stage3_holdout_check(
    *,
    rows: list[dict[str, Any]],
    input_path: str,
    base_workdir: str,
    checkpoint_every: int,
    capture_rationale: bool,
    temperature: float,
    max_new_tokens: int,
    prompt_method: str,
    winner_model: str,
    runner_up_model: str | None,
    validation_state_fidelity: float,
) -> dict[str, Any]:
    test_rows = select_split_rows(rows, "test")
    initial_result = _run_arm(
        rows=test_rows,
        input_path=input_path,
        base_workdir=base_workdir,
        stage_name="stage3_holdout",
        arm_name=winner_model,
        model=winner_model,
        prompt_method=prompt_method,
        checkpoint_every=checkpoint_every,
        capture_rationale=capture_rationale,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    initial_pass = stage3_holdout_passes(initial_result["summary"], validation_state_fidelity=validation_state_fidelity)
    final_result = initial_result
    rerun_result: dict[str, Any] | None = None
    if not initial_pass and runner_up_model and runner_up_model != winner_model:
        rerun_result = _run_arm(
            rows=test_rows,
            input_path=input_path,
            base_workdir=base_workdir,
            stage_name="stage3_holdout_rerun",
            arm_name=runner_up_model,
            model=runner_up_model,
            prompt_method=prompt_method,
            checkpoint_every=checkpoint_every,
            capture_rationale=capture_rationale,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        if stage3_holdout_passes(rerun_result["summary"], validation_state_fidelity=validation_state_fidelity):
            final_result = rerun_result

    stage_summary = {
        "stage": "stage3_holdout",
        "testRows": len(test_rows),
        "goldBackedTestRows": sum(1 for row in test_rows if str(row.get("gold_brief", "")).strip()),
        "promptMethod": prompt_method,
        "initialModel": winner_model,
        "runnerUpModel": runner_up_model,
        "chosenModel": final_result["model"],
        "pass": stage3_holdout_passes(final_result["summary"], validation_state_fidelity=validation_state_fidelity),
        "validationReferenceStateFidelity": validation_state_fidelity,
        "initialResult": initial_result,
        "rerunResult": rerun_result,
    }
    write_json(derive_stage_summary_path(base_workdir, "stage3_holdout"), stage_summary)
    return stage_summary

