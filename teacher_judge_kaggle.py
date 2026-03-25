from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
except ImportError:  # pragma: no cover - depends on Kaggle image
    BitsAndBytesConfig = None

import teacher_runtime as tr


DEFAULT_WORKDIR = "/kaggle/working/qwen_teacher_judge"
DEFAULT_OUTPUT_PATH = "/kaggle/working/qwen_teacher_judge_rows.json"
DEFAULT_SUMMARY_OUTPUT = "/kaggle/working/qwen_teacher_judge_summary.json"
DEFAULT_TEACHER_MODEL = tr.DEFAULT_MODEL
DEFAULT_TEACHER_QUANTIZATION = "fp16"
DEFAULT_TEACHER_GPU = 0
DEFAULT_TEACHER_MAX_NEW_TOKENS = 420
DEFAULT_TEACHER_TEMPERATURE = 0.2
DEFAULT_JUDGE_MODEL = "Qwen/Qwen2.5-14B-Instruct"
DEFAULT_JUDGE_QUANTIZATION = "4bit"
DEFAULT_JUDGE_GPU = 1
DEFAULT_JUDGE_MAX_NEW_TOKENS = 220
DEFAULT_JUDGE_TRANSCRIPT_CHAR_BUDGET = 10000
DEFAULT_JUDGE_TURN_CHAR_LIMIT = 1000
DEFAULT_CHECKPOINT_EVERY = 1

JUDGE_SCORE_FIELDS = (
    "overall_score",
    "faithfulness_score",
    "carry_forward_score",
    "next_step_score",
    "missing_context_score",
    "hallucination_risk_score",
    "composite_score",
)
JUDGE_TEXT_FIELDS = ("strengths", "issues")
JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)
THINK_BLOCK_PATTERN = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
FATAL_TEACHER_ISSUES = {
    "empty or near-empty normalized output",
    "prompt echo in opening text",
    "contains placeholder bullet",
    "obvious unusable scaffolding",
}
STRUCTURE_ONLY_TEACHER_ISSUES = {
    "section coverage below 4/8 required content headers",
    "missing both continuation tail anchors",
}


@dataclass
class JudgeOutcome:
    status: str
    verdict: str
    scores: dict[str, float]
    strengths: list[str]
    issues: list[str]
    recommended_fix: str
    raw_output: str
    duration_ms: int
    parse_error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quality-first Kaggle teacher+judge pipeline for continuation brief generation on dual T4 GPUs."
    )
    parser.add_argument("--run-dataset", default="full", choices=("full", "repair"))
    parser.add_argument("--input-path-override", default="")
    parser.add_argument("--existing-teacher-output", default="")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--workdir", default=DEFAULT_WORKDIR)
    parser.add_argument("--split", default="", help="Optional split filter: train, val, or test.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument("--prompt-method", default=tr.DEFAULT_PROMPT_METHOD, choices=tr.PROMPT_METHODS)
    parser.add_argument("--capture-rationale", action="store_true")
    parser.add_argument("--teacher-model", default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--teacher-quantization", default=DEFAULT_TEACHER_QUANTIZATION, choices=("fp16", "8bit", "4bit"))
    parser.add_argument("--teacher-gpu", type=int, default=DEFAULT_TEACHER_GPU)
    parser.add_argument("--teacher-temperature", type=float, default=DEFAULT_TEACHER_TEMPERATURE)
    parser.add_argument("--teacher-max-new-tokens", type=int, default=DEFAULT_TEACHER_MAX_NEW_TOKENS)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-quantization", default=DEFAULT_JUDGE_QUANTIZATION, choices=("fp16", "8bit", "4bit"))
    parser.add_argument("--judge-gpu", type=int, default=DEFAULT_JUDGE_GPU)
    parser.add_argument("--judge-max-new-tokens", type=int, default=DEFAULT_JUDGE_MAX_NEW_TOKENS)
    parser.add_argument("--judge-transcript-char-budget", type=int, default=DEFAULT_JUDGE_TRANSCRIPT_CHAR_BUDGET)
    parser.add_argument("--judge-turn-char-limit", type=int, default=DEFAULT_JUDGE_TURN_CHAR_LIMIT)
    return parser.parse_args()


def build_quantization_config(mode: str) -> Any | None:
    if mode == "fp16":
        return None
    if BitsAndBytesConfig is None:
        raise RuntimeError("Quantized loading requires transformers.BitsAndBytesConfig, but it is unavailable.")
    if mode == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    if mode == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    raise ValueError(f"Unsupported quantization mode: {mode}")


def tokenizer_kwargs() -> dict[str, Any]:
    token = os.getenv("HF_TOKEN", "").strip()
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "use_fast": False,
    }
    if token:
        kwargs["token"] = token
    return kwargs


def model_kwargs(model_id: str, quantization: str, device_index: int) -> dict[str, Any]:
    settings = tr.resolve_model_load_settings(device_index)
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": settings["torch_dtype"],
        "attn_implementation": settings["attn_implementation"],
        "low_cpu_mem_usage": True,
    }
    token = os.getenv("HF_TOKEN", "").strip()
    if token:
        kwargs["token"] = token
    quant_config = build_quantization_config(quantization)
    if quant_config is not None:
        kwargs["quantization_config"] = quant_config
        kwargs["device_map"] = {"": device_index}
    return kwargs


def strip_model_json(text: str) -> str:
    cleaned = THINK_BLOCK_PATTERN.sub("", text).strip()
    fenced = JSON_BLOCK_PATTERN.search(cleaned)
    if fenced:
        return fenced.group(1).strip()
    return cleaned


def extract_first_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    cleaned = strip_model_json(text)
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("Judge output did not contain a parseable JSON object.")


def clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(min(10.0, max(0.0, score)), 3)


def normalize_text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def normalize_verdict(value: Any) -> str:
    lowered = str(value or "").strip().lower()
    if lowered in {"accept", "pass", "good"}:
        return "accept"
    if lowered in {"review", "partial", "mixed"}:
        return "review"
    if lowered in {"reject", "fail", "poor", "bad"}:
        return "reject"
    return "review"


def compute_composite_score(scores: dict[str, float]) -> float:
    positive = (
        0.30 * scores["overall_score"]
        + 0.25 * scores["faithfulness_score"]
        + 0.20 * scores["carry_forward_score"]
        + 0.15 * scores["next_step_score"]
        + 0.05 * (10.0 - scores["missing_context_score"])
        + 0.05 * (10.0 - scores["hallucination_risk_score"])
    )
    return round(positive, 3)


def normalize_judge_payload(payload: dict[str, Any]) -> dict[str, Any]:
    scores = {
        "overall_score": clamp_score(payload.get("overall_score", payload.get("overall"))),
        "faithfulness_score": clamp_score(payload.get("faithfulness_score", payload.get("faithfulness"))),
        "carry_forward_score": clamp_score(payload.get("carry_forward_score", payload.get("carry_forward"))),
        "next_step_score": clamp_score(payload.get("next_step_score", payload.get("next_step"))),
        "missing_context_score": clamp_score(payload.get("missing_context_score", payload.get("missing_context"))),
        "hallucination_risk_score": clamp_score(
            payload.get("hallucination_risk_score", payload.get("hallucination_risk"))
        ),
    }
    scores["composite_score"] = compute_composite_score(scores)
    return {
        "verdict": normalize_verdict(payload.get("verdict", payload.get("overall_verdict"))),
        "scores": scores,
        "strengths": normalize_text_list(payload.get("strengths")),
        "issues": normalize_text_list(payload.get("issues")),
        "recommended_fix": str(payload.get("recommended_fix", payload.get("fix", ""))).strip(),
    }


class HFChatModel:
    def __init__(
        self,
        *,
        model_id: str,
        quantization: str,
        device_index: int,
        temperature: float,
        max_new_tokens: int,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this script.")
        if device_index < 0 or device_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested cuda:{device_index}, but only {torch.cuda.device_count()} CUDA device(s) are visible."
            )
        self.model_id = model_id
        self.quantization = quantization
        self.device_index = device_index
        self.device = torch.device(f"cuda:{device_index}")
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs())
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        load_kwargs = model_kwargs(model_id, quantization, device_index)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        except TypeError:
            load_kwargs.pop("attn_implementation", None)
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        if quantization == "fp16":
            self.model.to(self.device)
        self.model.eval()

    def _tokenize(self, prompt: str, *, enable_thinking: bool = False) -> dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict[str, Any] = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            if "qwen" in self.model_id.lower() and not enable_thinking:
                kwargs["enable_thinking"] = False
            try:
                inputs = self.tokenizer.apply_chat_template(messages, **kwargs)
            except TypeError:
                kwargs.pop("enable_thinking", None)
                inputs = self.tokenizer.apply_chat_template(messages, **kwargs)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt")
        return inputs.to(self.device)

    def generate(self, prompt: str, *, max_new_tokens: int | None = None, enable_thinking: bool = False) -> str:
        inputs = self._tokenize(prompt, enable_thinking=enable_thinking)
        generated = None
        new_tokens = None
        try:
            generation_kwargs: dict[str, Any] = {
                **inputs,
                "do_sample": self.temperature > 0,
                "max_new_tokens": max_new_tokens or self.max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if self.temperature > 0:
                generation_kwargs["temperature"] = max(self.temperature, 1e-5)
            with torch.inference_mode():
                generated = self.model.generate(**generation_kwargs)
            new_tokens = generated[:, inputs["input_ids"].shape[1] :]
            return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
        finally:
            del inputs
            if new_tokens is not None:
                del new_tokens
            if generated is not None:
                del generated
            tr.clear_torch_memory(self.device)

    def close(self) -> None:
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        gc.collect()
        tr.clear_torch_memory(self.device)


class TeacherGeneratorAdapter(HFChatModel):
    def generate(self, prompt: str, capture_rationale: bool, *, max_new_tokens: int | None = None) -> str:  # type: ignore[override]
        return super().generate(
            prompt,
            max_new_tokens=max_new_tokens,
            enable_thinking=bool(capture_rationale),
        )


def _teacher_issue_set(candidate_status: str, issues: list[str]) -> set[str]:
    if candidate_status != "hard_reject":
        return set()
    return {str(issue).strip() for issue in issues if str(issue).strip()}


def _teacher_candidate_is_usable(candidate_status: str, issues: list[str], brief: str) -> bool:
    if not brief.strip():
        return False
    issue_set = _teacher_issue_set(candidate_status, issues)
    if not issue_set:
        return True
    return not any(issue in FATAL_TEACHER_ISSUES for issue in issue_set)


def _quality_first_status(candidate_status: str, issues: list[str], brief: str) -> str:
    if candidate_status != "hard_reject":
        return candidate_status
    if _teacher_candidate_is_usable(candidate_status, issues, brief):
        return "soft_accept"
    return "hard_reject"


def generate_teacher_draft_quality_first(
    *,
    generator: TeacherGeneratorAdapter,
    row: dict[str, Any],
    capture_rationale: bool,
    prompt_method: str,
) -> tr.GenerationOutcome:
    attempts = tr.build_generation_attempts(generator.max_new_tokens)
    last_oom_error: RuntimeError | None = None
    started_at = time.time()
    attempt_debug_records: list[dict[str, Any]] = []
    best_outcome: tr.GenerationOutcome | None = None

    for attempt_index, attempt in enumerate(attempts, start=1):
        prompt = tr.build_prompt(
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
            brief_text, auxiliary_rationale = tr.split_model_output(raw_output)
            brief = tr.normalize_generated_brief(brief_text)
            candidate_status, issues = tr.classify_generated_brief(brief)
            chosen_status = _quality_first_status(candidate_status, issues, brief)

            debug_record = {
                "attemptIndex": attempt_index,
                "transcriptCharBudget": attempt["transcript_char_budget"],
                "maxNewTokens": attempt["max_new_tokens"],
                "promptLength": len(prompt),
                "promptPreview": tr.trim_debug_text(prompt),
                "rawOutput": tr.trim_debug_text(raw_output),
                "normalizedBrief": tr.trim_debug_text(brief),
                "structureStatus": candidate_status,
                "structureIssues": list(issues),
                "chosenStatus": chosen_status,
            }
            attempt_debug_records.append(debug_record)

            current_outcome = tr.GenerationOutcome(
                brief=brief,
                raw_output=raw_output,
                auxiliary_rationale=auxiliary_rationale,
                status=chosen_status,
                issues=list(issues),
                debug_payload={
                    "attempts": list(attempt_debug_records),
                    "chosenStatus": chosen_status,
                    "chosenIssues": list(issues),
                    "structureStatus": candidate_status,
                    "structureIssues": list(issues),
                },
                duration_ms=round((time.time() - started_at) * 1000),
            )

            if best_outcome is None or len(brief) > len(best_outcome.brief):
                best_outcome = current_outcome

            if chosen_status != "hard_reject":
                if candidate_status != "format_clean":
                    print(
                        f"quality-first accept for '{row.get('title', '')}' on attempt {attempt_index}/{len(attempts)}: "
                        f"structure_status={candidate_status}; {tr.summarize_issues(issues)}",
                        flush=True,
                    )
                return current_outcome

            print(
                f"quality-first retry for '{row.get('title', '')}' on attempt {attempt_index}/{len(attempts)}: "
                f"{tr.summarize_issues(issues)}",
                flush=True,
            )
        except RuntimeError as exc:
            if not tr.is_cuda_oom_error(exc):
                raise
            last_oom_error = exc
            print(
                "OOM while generating "
                f"'{row.get('title', '')}' on attempt {attempt_index}/{len(attempts)}; "
                f"retrying with transcript_char_budget={attempt['transcript_char_budget']} "
                f"and max_new_tokens={attempt['max_new_tokens']}",
                flush=True,
            )
            tr.clear_torch_memory(generator.device)

    if best_outcome is not None:
        return best_outcome
    if last_oom_error is not None:
        raise RuntimeError(f"{last_oom_error} after {len(attempts)} memory-recovery attempts.") from last_oom_error
    raise RuntimeError("Teacher draft generation exhausted all attempts without producing output.")


class JudgeModel(HFChatModel):
    def __init__(self, *, model_id: str, quantization: str, device_index: int, max_new_tokens: int) -> None:
        super().__init__(
            model_id=model_id,
            quantization=quantization,
            device_index=device_index,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
        )

    def judge(self, row: dict[str, Any], candidate_brief: str, *, transcript_char_budget: int, per_turn_char_limit: int) -> JudgeOutcome:
        prompt = build_judge_prompt(
            row,
            candidate_brief,
            transcript_char_budget=transcript_char_budget,
            per_turn_char_limit=per_turn_char_limit,
        )
        started_at = time.time()
        first_raw = self.generate(prompt, max_new_tokens=self.max_new_tokens)
        try:
            parsed = normalize_judge_payload(extract_first_json_object(first_raw))
            return JudgeOutcome(
                status="judged",
                verdict=parsed["verdict"],
                scores=parsed["scores"],
                strengths=parsed["strengths"],
                issues=parsed["issues"],
                recommended_fix=parsed["recommended_fix"],
                raw_output=first_raw,
                duration_ms=round((time.time() - started_at) * 1000),
            )
        except Exception as first_error:  # noqa: BLE001
            retry_prompt = (
                f"{prompt}\n\nYour previous answer was not valid JSON. "
                "Regenerate and return only one JSON object with the required keys and numeric scores."
            )
            retry_raw = self.generate(retry_prompt, max_new_tokens=self.max_new_tokens)
            try:
                parsed = normalize_judge_payload(extract_first_json_object(retry_raw))
                return JudgeOutcome(
                    status="judged",
                    verdict=parsed["verdict"],
                    scores=parsed["scores"],
                    strengths=parsed["strengths"],
                    issues=parsed["issues"],
                    recommended_fix=parsed["recommended_fix"],
                    raw_output=retry_raw,
                    duration_ms=round((time.time() - started_at) * 1000),
                )
            except Exception as retry_error:  # noqa: BLE001
                return JudgeOutcome(
                    status="judge_parse_error",
                    verdict="review",
                    scores={name: 0.0 for name in JUDGE_SCORE_FIELDS},
                    strengths=[],
                    issues=[],
                    recommended_fix="",
                    raw_output=retry_raw,
                    duration_ms=round((time.time() - started_at) * 1000),
                    parse_error=f"{type(first_error).__name__}: {first_error}; {type(retry_error).__name__}: {retry_error}",
                )


def render_judge_state(row: dict[str, Any]) -> str:
    state = row.get("rolling_state") or {}
    sections = [
        tr.render_list(
            "Objective",
            [tr.sanitize_text(state.get("objective"))] if tr.sanitize_text(state.get("objective")) else [],
            "Objective still needs confirmation.",
        ),
        tr.render_list(
            "Established context",
            [tr.sanitize_text(item) for item in (state.get("constraints") or []) if tr.sanitize_text(item)],
            "No explicit constraints have been captured yet.",
        ),
        tr.render_list(
            "Decisions already made",
            [tr.sanitize_text(item) for item in (state.get("decisions") or []) if tr.sanitize_text(item)],
            "No firm decisions are recorded yet.",
        ),
        tr.render_list(
            "Rejected paths / do not revisit unless necessary",
            [tr.sanitize_text(item) for item in (state.get("rejected") or []) if tr.sanitize_text(item)],
            "No rejected paths have been recorded yet.",
        ),
        tr.render_list(
            "Open questions",
            [tr.sanitize_text(item) for item in (state.get("open_questions") or []) if tr.sanitize_text(item)],
            "No open questions are active right now.",
        ),
        tr.render_list(
            "Where we left off",
            [tr.sanitize_text(state.get("next_step"))] if tr.sanitize_text(state.get("next_step")) else [],
            "Resume from the latest visible turn.",
        ),
    ]
    return "\n\n".join(sections)


def build_judge_prompt(
    row: dict[str, Any],
    candidate_brief: str,
    *,
    transcript_char_budget: int,
    per_turn_char_limit: int,
) -> str:
    transcript_text = tr.build_transcript_text(
        row.get("transcript_turns") or [],
        per_turn_char_limit=per_turn_char_limit,
        total_char_budget=transcript_char_budget,
    )
    return "\n".join(
        [
            "You are grading a continuation brief for a later chat handoff.",
            "Focus on content quality, not whether the exact section template is perfect.",
            "Use the transcript as the source of truth. Use the rolling state only as a helper.",
            "Penalize invented facts, dropped decisions, lost constraints, weak next-step handoff, and missing critical context.",
            "Return only one JSON object with these keys:",
            "{",
            '  "overall_score": 0-10 number,',
            '  "faithfulness_score": 0-10 number,',
            '  "carry_forward_score": 0-10 number,',
            '  "next_step_score": 0-10 number,',
            '  "missing_context_score": 0-10 number,',
            '  "hallucination_risk_score": 0-10 number,',
            '  "verdict": "accept" | "review" | "reject",',
            '  "strengths": ["..."],',
            '  "issues": ["..."],',
            '  "recommended_fix": "..."',
            "}",
            "",
            "Conversation metadata:",
            f"- title: {row.get('title', '')}",
            f"- provider: {row.get('provider', '')}",
            f"- split: {row.get('split', '')}",
            "",
            "Transcript excerpt:",
            transcript_text or "No transcript turns were provided.",
            "",
            "Rolling state helper:",
            render_judge_state(row),
            "",
            "Candidate continuation brief:",
            candidate_brief,
        ]
    )


def apply_judge_metadata(
    row: dict[str, Any],
    *,
    outcome: JudgeOutcome,
    judge_model: str,
    judge_quantization: str,
) -> dict[str, Any]:
    next_row = dict(row)
    next_row["teacher_judge_status"] = outcome.status
    next_row["teacher_judge_model"] = judge_model
    next_row["teacher_judge_quantization"] = judge_quantization
    next_row["teacher_judge_duration_ms"] = outcome.duration_ms
    next_row["teacher_judge_verdict"] = outcome.verdict
    next_row["teacher_judge_scores"] = dict(outcome.scores)
    next_row["teacher_judge_strengths"] = list(outcome.strengths)
    next_row["teacher_judge_issues"] = list(outcome.issues)
    next_row["teacher_judge_recommended_fix"] = outcome.recommended_fix
    next_row["teacher_judge_raw_output"] = outcome.raw_output
    if outcome.parse_error:
        next_row["teacher_judge_parse_error"] = outcome.parse_error
    else:
        next_row.pop("teacher_judge_parse_error", None)
    return next_row


def candidate_brief_for_judge(row: dict[str, Any]) -> str:
    for key in ("teacher_normalized_candidate_brief", "teacher_draft_brief"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def quality_first_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    remapped: list[dict[str, Any]] = []
    for row in rows:
        cloned = dict(row)
        if not str(cloned.get("teacher_draft_brief", "")).strip():
            normalized = str(cloned.get("teacher_normalized_candidate_brief", "")).strip()
            if normalized:
                cloned["teacher_draft_brief"] = normalized
        remapped.append(cloned)
    return remapped


def mean_dict(rows: list[dict[str, Any]], field: str) -> float:
    values = [float(row.get(field, 0)) for row in rows if row.get(field) is not None]
    if not values:
        return 0.0
    return round(mean(values), 3)


def build_judge_aggregate(rows: list[dict[str, Any]], *, require_gold: bool = False) -> dict[str, Any]:
    judged = [
        row
        for row in rows
        if row.get("teacher_judge_status") == "judged"
        and (not require_gold or str(row.get("gold_brief", "")).strip())
    ]
    if not judged:
        return {"count": 0}
    verdict_counts = Counter(str(row.get("teacher_judge_verdict", "review")) for row in judged)
    score_payload = {
        score_name: round(
            mean(float((row.get("teacher_judge_scores") or {}).get(score_name, 0.0)) for row in judged),
            3,
        )
        for score_name in JUDGE_SCORE_FIELDS
    }
    return {
        "count": len(judged),
        "verdictCounts": dict(sorted(verdict_counts.items())),
        **score_payload,
    }


def build_summary(
    *,
    args: argparse.Namespace,
    input_path: str,
    requested_rows: int,
    rows: list[dict[str, Any]],
    started_at: float,
    status: str,
) -> dict[str, Any]:
    teacher_status_counts = Counter(str(row.get("teacher_generation_status", "")) for row in rows if row.get("teacher_generation_status"))
    teacher_structure_status_counts = Counter(
        str(row.get("teacher_structure_status", "")) for row in rows if row.get("teacher_structure_status")
    )
    judge_status_counts = Counter(str(row.get("teacher_judge_status", "")) for row in rows if row.get("teacher_judge_status"))
    quality_rows = quality_first_metric_rows(rows)
    summary = {
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "durationSeconds": round(time.time() - started_at, 3),
        "status": status,
        "inputPath": input_path,
        "existingTeacherOutput": args.existing_teacher_output or None,
        "workdir": args.workdir,
        "runDataset": args.run_dataset,
        "split": args.split or None,
        "requestedRows": requested_rows,
        "completedRows": len(rows),
        "teacherModel": args.teacher_model,
        "teacherQuantization": args.teacher_quantization,
        "teacherGpu": args.teacher_gpu,
        "judgeModel": args.judge_model,
        "judgeQuantization": args.judge_quantization,
        "judgeGpu": args.judge_gpu,
        "promptMethod": args.prompt_method,
        "teacherStatusCounts": dict(sorted(teacher_status_counts.items())),
        "teacherStructureStatusCounts": dict(sorted(teacher_structure_status_counts.items())),
        "judgeStatusCounts": dict(sorted(judge_status_counts.items())),
        "teacherAverageDurationMs": mean_dict(rows, "teacher_generation_duration_ms"),
        "judgeAverageDurationMs": mean_dict(rows, "teacher_judge_duration_ms"),
        "teacherGoldBackedMetricsQualityFirst": tr.aggregate_gold_metrics(quality_rows),
        "judgeAggregate": build_judge_aggregate(rows),
        "judgeGoldBackedAggregate": build_judge_aggregate(rows, require_gold=True),
    }
    return summary


def append_debug_record(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def select_rows_for_run(rows: list[dict[str, Any]], *, split: str, limit: int) -> list[dict[str, Any]]:
    selected = tr.select_split_rows(rows, split) if split else list(rows)
    return tr.select_rows(selected, limit)


def run_pipeline(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_path = (
        str(Path(args.existing_teacher_output).resolve())
        if args.existing_teacher_output
        else tr.resolve_input_path(args.run_dataset, args.input_path_override)
    )
    source_rows = tr.load_json_rows(input_path)
    selected_rows = select_rows_for_run(source_rows, split=args.split.strip(), limit=args.limit)
    started_at = time.time()
    output_path = Path(args.output_path)
    summary_path = Path(args.summary_output)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    judge_failure_path = workdir / "judge_failures.ndjson"
    teacher_failure_path = workdir / "teacher_failures.ndjson"

    generated_rows: list[dict[str, Any]]
    resumed_rows = 0
    if args.resume and output_path.exists():
        generated_rows = tr.load_json_rows(str(output_path))
        resumed_rows = len(generated_rows)
        if resumed_rows > len(selected_rows):
            raise RuntimeError("Existing output has more rows than the current selected dataset slice.")
        print(f"Resuming from row {resumed_rows + 1}/{len(selected_rows)}", flush=True)
    else:
        generated_rows = []
        for path in (output_path, summary_path, judge_failure_path, teacher_failure_path):
            if path.exists():
                path.unlink()

    teacher: TeacherGeneratorAdapter | None = None
    judge: JudgeModel | None = None
    try:
        if not args.existing_teacher_output:
            teacher = TeacherGeneratorAdapter(
                model_id=args.teacher_model,
                quantization=args.teacher_quantization,
                device_index=args.teacher_gpu,
                temperature=args.teacher_temperature,
                max_new_tokens=args.teacher_max_new_tokens,
            )
        judge = JudgeModel(
            model_id=args.judge_model,
            quantization=args.judge_quantization,
            device_index=args.judge_gpu,
            max_new_tokens=args.judge_max_new_tokens,
        )

        for index, row in enumerate(selected_rows[resumed_rows:], start=resumed_rows + 1):
            print(
                f"Processing {index}/{len(selected_rows)} "
                f"[teacher={args.teacher_model if teacher else 'existing'}] "
                f"[judge={args.judge_model}] {row.get('title', '')}",
                flush=True,
            )
            next_row = dict(row)
            if teacher is not None:
                try:
                    outcome = generate_teacher_draft_quality_first(
                        generator=teacher,
                        row=row,
                        capture_rationale=args.capture_rationale,
                        prompt_method=args.prompt_method,
                    )
                    next_row = tr.apply_generation_metadata(
                        row,
                        status=outcome.status,
                        issues=outcome.issues,
                        model=args.teacher_model,
                        prompt_method=args.prompt_method,
                        duration_ms=outcome.duration_ms,
                        auxiliary_rationale=outcome.auxiliary_rationale,
                        brief=outcome.brief,
                        raw_output=outcome.raw_output,
                    )
                    next_row["teacher_structure_status"] = str(outcome.debug_payload.get("structureStatus", outcome.status))
                    next_row["teacher_structure_issues"] = list(
                        outcome.debug_payload.get("structureIssues", outcome.issues)
                    )
                    if outcome.status == "hard_reject":
                        append_debug_record(
                            teacher_failure_path,
                            {
                                "title": row.get("title", ""),
                                "conversation_id": row.get("conversation_id", ""),
                                "message": "; ".join(outcome.issues),
                                **outcome.debug_payload,
                            },
                        )
                except Exception as exc:  # noqa: BLE001
                    next_row = tr.apply_generation_metadata(
                        row,
                        status="hard_reject",
                        issues=[f"runtime/model exception: {type(exc).__name__}", str(exc)],
                        model=args.teacher_model,
                        prompt_method=args.prompt_method,
                        duration_ms=0,
                    )
                    append_debug_record(
                        teacher_failure_path,
                        {
                            "title": row.get("title", ""),
                            "conversation_id": row.get("conversation_id", ""),
                            "errorType": type(exc).__name__,
                            "message": str(exc),
                        },
                    )

            candidate_brief = candidate_brief_for_judge(next_row)
            if candidate_brief:
                try:
                    judge_outcome = judge.judge(
                        next_row,
                        candidate_brief,
                        transcript_char_budget=args.judge_transcript_char_budget,
                        per_turn_char_limit=args.judge_turn_char_limit,
                    )
                except Exception as exc:  # noqa: BLE001
                    judge_outcome = JudgeOutcome(
                        status="judge_runtime_error",
                        verdict="review",
                        scores={name: 0.0 for name in JUDGE_SCORE_FIELDS},
                        strengths=[],
                        issues=[],
                        recommended_fix="",
                        raw_output="",
                        duration_ms=0,
                        parse_error=f"{type(exc).__name__}: {exc}",
                    )
                next_row = apply_judge_metadata(
                    next_row,
                    outcome=judge_outcome,
                    judge_model=args.judge_model,
                    judge_quantization=args.judge_quantization,
                )
                if judge_outcome.status != "judged":
                    append_debug_record(
                        judge_failure_path,
                        {
                            "title": row.get("title", ""),
                            "conversation_id": row.get("conversation_id", ""),
                            "status": judge_outcome.status,
                            "message": judge_outcome.parse_error,
                            "raw_output": tr.trim_debug_text(judge_outcome.raw_output),
                        },
                    )
            else:
                next_row = apply_judge_metadata(
                    next_row,
                    outcome=JudgeOutcome(
                        status="skipped_no_candidate",
                        verdict="review",
                        scores={name: 0.0 for name in JUDGE_SCORE_FIELDS},
                        strengths=[],
                        issues=[],
                        recommended_fix="",
                        raw_output="",
                        duration_ms=0,
                    ),
                    judge_model=args.judge_model,
                    judge_quantization=args.judge_quantization,
                )

            generated_rows.append(next_row)
            if len(generated_rows) % max(args.checkpoint_every, 1) == 0 or index == len(selected_rows):
                summary = build_summary(
                    args=args,
                    input_path=input_path,
                    requested_rows=len(selected_rows),
                    rows=generated_rows,
                    started_at=started_at,
                    status="running" if index < len(selected_rows) else "completed",
                )
                tr.write_json(output_path, generated_rows)
                tr.write_json(summary_path, summary)
    finally:
        if judge is not None:
            judge.close()
        if teacher is not None:
            teacher.close()

    final_summary = build_summary(
        args=args,
        input_path=input_path,
        requested_rows=len(selected_rows),
        rows=generated_rows,
        started_at=started_at,
        status="completed",
    )
    tr.write_json(output_path, generated_rows)
    tr.write_json(summary_path, final_summary)
    return generated_rows, final_summary


def main() -> int:
    args = parse_args()
    _, summary = run_pipeline(args)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
