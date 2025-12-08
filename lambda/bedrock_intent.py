"""
EdgeCoach AI - Intent classifier

This module calls Claude via Bedrock to extract interview intent signals
and enrich them with heuristics. It:
- Builds a structured prompt for company, role, timing, and intent.
- Parses JSON output safely and clamps missing fields.
- Falls back to regex/heuristic extraction when Bedrock fails.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import boto3

MODEL_ID = os.getenv(
    "BEDROCK_INTENT_MODEL_ID",
    "anthropic.claude-3-haiku-20240307-v1:0",
)

bedrock = boto3.client("bedrock-runtime")
logger = logging.getLogger(__name__)

DEFAULT_INTENT: Dict[str, Any] = {
    "company": "",
    "role": "",
    "time_to_interview_hours": 24,
    "level": "",
    "location": "",
    "wants_interview_intel": False,
}

_MIN_HOURS = 1
_MAX_HOURS = 336
_MAX_HISTORY_TURNS = 10
_ROLE_HINTS = {
    "analyst",
    "associate",
    "architect",
    "consultant",
    "coordinator",
    "designer",
    "developer",
    "director",
    "engineer",
    "intern",
    "lead",
    "manager",
    "marketer",
    "owner",
    "pm",
    "principal",
    "product manager",
    "program manager",
    "project manager",
    "recruiter",
    "representative",
    "researcher",
    "scientist",
    "specialist",
    "strategist",
}
_PHRASE_TO_HOURS: List[Tuple[str, int]] = [
    ("day after tomorrow", 48),
    ("day-after-tomorrow", 48),
    ("later today", 12),
    ("today", 12),
    ("tonight", 12),
    ("tomorrow", 24),
    ("this weekend", 72),
    ("weekend", 72),
    ("early next week", 120),
    ("next week", 168),
    ("in a week", 168),
    ("two weeks", 336),
    ("2 weeks", 336),
    ("fortnight", 336),
    ("next month", 336),
]
_IN_HOURS_RE = re.compile(r"in\s+(\d+)\s+(?:hours?|hrs?)", re.IGNORECASE)
_IN_DAYS_RE = re.compile(r"in\s+(\d+)\s+days?", re.IGNORECASE)
_IN_WEEKS_RE = re.compile(r"in\s+(\d+)\s+weeks?", re.IGNORECASE)
_COMPANY_TOKEN = r"[A-Z][A-Za-z0-9&./+-]*"
_COMPANY_BODY = rf"{_COMPANY_TOKEN}(?:\s+{_COMPANY_TOKEN}){{0,3}}"
_ROLE_BODY = r"[A-Za-z][A-Za-z0-9/&+ .-]{0,80}"
_COMPANY_ROLE_PATTERNS = [
    re.compile(
        rf"\bhave\s+(?:an|a)\s+(?P<company>{_COMPANY_BODY})\s+(?P<role>{_ROLE_BODY})\s+interview",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\binterview\s+with\s+(?P<company>{_COMPANY_BODY})(?:[^.?!]*?\bfor\s+(?P<role>{_ROLE_BODY}))?",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\binterview\s+at\s+(?P<company>{_COMPANY_BODY})(?:[^.?!]*?\bfor\s+(?P<role>{_ROLE_BODY}))?",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?P<company>{_COMPANY_BODY})\s+(?P<role>{_ROLE_BODY})\s+interview",
        re.IGNORECASE,
    ),
]

_SYSTEM_PROMPT = """
You are IntentJSON, a deterministic classification service operating in strict JSON mode.

Contract:
1. Always respond with exactly one JSON object and nothing else (no markdown, code fences, narration, or trailing commas).
2. The JSON object MUST include the following keys and value types:
   - "company": string (use "" if unknown)
   - "role": string (use "" if unknown)
   - "time_to_interview_hours": integer between 1 and 336
   - "level": string (use "" if not provided)
   - "location": string (use "" if not provided)
   - "wants_interview_intel": boolean
3. Do not invent facts. Extract only from the conversation.
4. wants_interview_intel = true when the user asks about interview questions, tips, what to expect, or insider knowledge.
5. Convert all temporal language into hours:
   - "later today", "today", or "tonight" => 12
   - "tomorrow" => 24
   - "day after tomorrow" => 48
   - "this weekend" => 72
   - "early next week" => 120
   - "next week" => 168
   - "two weeks", "fortnight" => 336
   - "next month" or anything longer => 336
   - "in X hours" => X
   - "in X days" => X * 24
   - "in X weeks" => X * 168
   - If a precise datetime is given, compute the approximate hour delta (clamp to 1-336).
6. When time is unspecified, default to 24 hours.

The JSON response must strictly follow:
{
  "company": "",
  "role": "",
  "time_to_interview_hours": 24,
  "level": "",
  "location": "",
  "wants_interview_intel": false
}
""".strip()


def extract_intent(user_query: str, history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Calls Claude Haiku 4.5 on Bedrock to extract interview-intent signals.
    Returns DEFAULT_INTENT if the model output is missing or invalid.
    """

    user_prompt = _build_user_prompt(user_query, history)

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 256,
        "temperature": 0,
        "system": _SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ],
    }

    try:
        response = bedrock.invoke_model(
            modelId=MODEL_ID,
            accept="application/json",
            contentType="application/json",
            body=json.dumps(payload),
        )
        raw_text = _extract_model_text(response)
        intent = _parse_intent_json(raw_text)
    except Exception as exc:
        logger.warning("Claude intent extraction failed; using heuristic fallback: %s", exc)
        intent = _fallback_intent(user_query, history)

    return _enrich_with_context(intent, user_query, history)


def _build_user_prompt(user_query: str, history: Optional[List[Dict[str, Any]]]) -> str:
    history_block = _format_history(history)
    return (
        "Conversation history (oldest to newest):\n"
        f"{history_block}\n\n"
        "Latest user message:\n"
        f"{user_query.strip()}\n\n"
        "Return ONLY the JSON object described in the system prompt."
    )


def _format_history(history: Optional[List[Dict[str, Any]]]) -> str:
    if not history:
        return "None."

    trimmed = history[-_MAX_HISTORY_TURNS:]
    lines: List[str] = []
    for turn in trimmed:
        role = str(turn.get("role", "UNKNOWN")).upper()
        content = _stringify_content(turn.get("content", ""))
        lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "None."


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(_stringify_content(item) for item in content)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _extract_model_text(response: Dict[str, Any]) -> str:
    body = response.get("body")
    body_bytes = body.read() if hasattr(body, "read") else body
    if isinstance(body_bytes, (bytes, bytearray)):
        body_bytes = body_bytes.decode("utf-8")

    payload = json.loads(body_bytes or "{}")
    content_blocks = payload.get("content") or []
    text_chunks = [
        block.get("text", "")
        for block in content_blocks
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    text = "".join(text_chunks).strip()
    if not text:
        raise ValueError("Empty model output")
    return text


def _parse_intent_json(raw_text: str) -> Dict[str, Any]:
    data = json.loads(raw_text)
    if not isinstance(data, dict):
        raise ValueError("Model output is not a JSON object")

    return {
        "company": _coerce_str(data.get("company")),
        "role": _coerce_str(data.get("role")),
        "time_to_interview_hours": _coerce_hours(
            data.get("time_to_interview_hours"), DEFAULT_INTENT["time_to_interview_hours"]
        ),
        "level": _coerce_str(data.get("level")),
        "location": _coerce_str(data.get("location")),
        "wants_interview_intel": _coerce_bool(
            data.get("wants_interview_intel"), DEFAULT_INTENT["wants_interview_intel"]
        ),
    }


def _fallback_intent(user_query: str, history: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    blob_all = _compose_text_blob(user_query, history)
    blob_current = user_query or ""

    company, role = _guess_company_role(blob_all)

    fallback = DEFAULT_INTENT.copy()
    if company:
        fallback["company"] = company
    if role:
        fallback["role"] = role

    fallback["time_to_interview_hours"] = _infer_time_hours(blob_all)
    fallback["wants_interview_intel"] = _infer_wants_interview_intel(blob_current)

    return fallback




def _enrich_with_context(intent: Dict[str, Any], user_query: str, history: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    enriched = DEFAULT_INTENT.copy()
    enriched.update(intent or {})

    blob_all = _compose_text_blob(user_query, history)
    blob_current = user_query or ""

    company, role = _guess_company_role(blob_all)

    if not enriched["company"] and company:
        enriched["company"] = company
    if not enriched["role"] and role:
        enriched["role"] = role
    if not enriched["time_to_interview_hours"]:
        enriched["time_to_interview_hours"] = _infer_time_hours(blob_all)

    if not enriched["wants_interview_intel"] and _infer_wants_interview_intel(blob_current):
        enriched["wants_interview_intel"] = True

    enriched["time_to_interview_hours"] = _coerce_hours(
        enriched.get("time_to_interview_hours"),
        DEFAULT_INTENT["time_to_interview_hours"],
    )

    return enriched




def _compose_text_blob(user_query: str, history: Optional[List[Dict[str, Any]]]) -> str:
    parts: List[str] = []
    if history:
        for turn in history[-_MAX_HISTORY_TURNS:]:
            parts.append(_stringify_content(turn.get("content", "")))
    if user_query:
        parts.append(user_query)
    return " ".join(part for part in parts if part).strip()


def _coerce_str(value: Any, default: str = "") -> str:
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned if cleaned else default
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "t", "yes", "y", "1"}:
            return True
        if lowered in {"false", "f", "no", "n", "0"}:
            return False
    return default


def _coerce_hours(value: Any, default: int) -> int:
    if isinstance(value, bool) or value is None:
        return default
    try:
        hours = int(float(value))
    except (TypeError, ValueError):
        return default
    return _clamp_hours(hours)


def _guess_company_role(text: str) -> Tuple[str, str]:
    best_company = ""
    best_role = ""
    for pattern in _COMPANY_ROLE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        company = _clean_company(match.group("company") or "")
        role = _clean_role(match.group("role") or "")
        if company and not best_company:
            best_company = company
        if role and _looks_like_role(role):
            if not best_role:
                best_role = role
            resolved_company = company or best_company
            if resolved_company:
                return resolved_company, role
    return best_company, best_role


def _clean_company(candidate: str) -> str:
    cleaned = candidate.strip(" .,!?:;-")
    return cleaned


def _clean_role(candidate: str) -> str:
    role = (candidate or "").strip(" .,!?:;-")
    role = re.sub(r"\b(role|position|job)\b", "", role, flags=re.IGNORECASE)
    role = re.sub(r"\s+", " ", role).strip()
    return role


def _looks_like_role(role: str) -> bool:
    if not role:
        return False
    role_lower = role.lower()
    return any(hint in role_lower for hint in _ROLE_HINTS)


def _infer_time_hours(text: str) -> int:
    lowered = text.lower()
    for phrase, hours in _PHRASE_TO_HOURS:
        if phrase in lowered:
            return hours

    match = _IN_HOURS_RE.search(lowered)
    if match:
        return _clamp_hours(int(match.group(1)))

    match = _IN_DAYS_RE.search(lowered)
    if match:
        return _clamp_hours(int(match.group(1)) * 24)

    match = _IN_WEEKS_RE.search(lowered)
    if match:
        return _clamp_hours(int(match.group(1)) * 168)

    return DEFAULT_INTENT["time_to_interview_hours"]


def _infer_wants_interview_intel(text: str) -> bool:
    lowered = text.lower()
    keywords = ["interview", "intel", "tips", "questions", "process", "prep", "coach", "brief"]
    return any(keyword in lowered for keyword in keywords)


def _clamp_hours(value: int) -> int:
    return max(_MIN_HOURS, min(_MAX_HOURS, value))
