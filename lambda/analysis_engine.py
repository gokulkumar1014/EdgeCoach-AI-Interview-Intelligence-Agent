"""
EdgeCoach AI - RAG synthesis

This module builds the interview synthesis prompt and calls Bedrock. It:
- Formats sources into context blocks and composes the user prompt.
- Invokes Claude to generate prep guidance and answer markdown.
- Provides defaults when sources are missing or model calls fail.
"""
from typing import List, Dict, Any, Tuple
import json
import logging
import os

import boto3

MODEL_ID = os.getenv(
    "BEDROCK_ANALYSIS_MODEL_ID",
    "anthropic.claude-3-haiku-20240307-v1:0",
)

bedrock = boto3.client("bedrock-runtime")
logger = logging.getLogger(__name__)

MAX_SOURCES = 3
MAX_SOURCE_CONTENT = 2500
MAX_CONTEXT = 9000
MAX_HISTORY_TURNS = 8


def synthesize_interview_answer(
    user_query: str,
    intent: Dict[str, Any],
    sources: List[Dict[str, Any]],
    messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    formatted_sources, context_block = _prepare_sources(sources)
    if not formatted_sources:
        return _default_response([])

    company = (intent.get("company") or "Unknown company").strip() or "Unknown company"
    role = (intent.get("role") or "Unknown role").strip() or "Unknown role"
    time_hours = _coerce_int(intent.get("time_to_interview_hours"), 24)

    conversation_snippet = _format_history(messages)
    time_guidance = _time_window_guidance(time_hours)
    user_prompt = _compose_user_prompt(
        company=company,
        role=role,
        time_hours=time_hours,
        time_guidance=time_guidance,
        conversation_snippet=conversation_snippet,
        context_block=context_block,
        user_query=user_query,
    )

    payload = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 4000,
    "temperature": 0.5,
    "system": """You are EdgeCoach AI — an energetic, encouraging, and highly skilled career mentor and interview coach. You speak with warmth, clarity, and confidence.

                Your core style:
                - Positive, motivating, and supportive.
                - Clear, structured explanations.
                - Teacher-like when the user asks any concept (ML, coding, stats, analytics,
                math, business, anything).
                - Interview-coach mode when the question relates to hiring, interviews,
                job roles, or career prep.
                - Never overwhelm — keep guidance practical and focused.
                - No inline citations like [S1] or weird source markers.

                When handling interview intel:
                - Deliver step-by-step insights, examples, and tailored preparation plans.
                - Combine empathy (“It's normal to feel nervous…”) with actionable advice.
                - Turn raw source context into clean, structured insights (flow, rounds,
                themes, prep plan, takeaways).

                When handling general knowledge:
                - Teach concepts clearly, with simple analogies and examples.
                - Maintain the same friendly, coach-like tone.

                Always:
                - Follow the format requested in the user prompt.
                - Be concise but impactful.
                - Reassure the user and help them grow in confidence.""",
    "messages": [
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
    ],
}


    try:
        response_json = _invoke_bedrock(payload)
        answer_markdown = _collect_text(response_json)
        if not answer_markdown:
            raise ValueError("Claude returned no content")
        return {
            "answer_markdown": answer_markdown,
            "sections": [],
            "sources": formatted_sources,
        }
    except Exception as exc:
        logger.exception("Interview synthesis failed: %s", exc)
        return _default_response(formatted_sources)


def _prepare_sources(sources: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
    formatted: List[Dict[str, Any]] = []
    context_blocks: List[str] = []

    for doc in sources or []:
        if len(formatted) >= MAX_SOURCES:
            break

        content = (doc.get("content") or "").strip()
        snippet = (doc.get("snippet") or doc.get("content") or "")[:300]
        if not content:
            continue

        doc_id = doc.get("id") or f"S{len(formatted) + 1}"
        title = (doc.get("title") or doc.get("url") or "Untitled source").strip()
        source_domain = (doc.get("source") or "web").strip()
        url = doc.get("url")

        formatted.append(
            {
                "id": doc_id,
                "title": title,
                "url": url,
                "source": source_domain,
                "snippet": snippet,
            }
        )

        context_blocks.append(
            f"[{doc_id}] {title}\n"
            f"Source: {source_domain}\n"
            f"URL: {url or 'unknown'}\n"
            f"Content:\n{content[:MAX_SOURCE_CONTENT]}\n"
        )

    context = "\n\n".join(context_blocks).strip()[:MAX_CONTEXT]
    return formatted, context


def _compose_user_prompt(
    company: str,
    role: str,
    time_hours: int,
    time_guidance: str,
    conversation_snippet: str,
    context_block: str,
    user_query: str,
) -> str:
    return f"""
Candidate profile:
- Company: {company}
- Role: {role}
- Time to interview: {time_hours} hours
- Guidance: {time_guidance}

Conversation summary:
{conversation_snippet}

SOURCE CONTEXT (trimmed):
{context_block}

OUTPUT FORMAT (no extra sections, no citations):

# 1. OVERALL SUMMARY
(set expectations, reassure anxious candidates)

# 2. INTERVIEW FLOW & ROUNDS
(bullets describing the sequence of screens or loops)

# 3. QUESTION THEMES & EXAMPLES
(grouped themes with example prompts referencing the sources)

# 4. PREP PLAN
(chronological actions tailored to remaining hours)

# 5. FINAL TAKEAWAYS
(confidence-building close)

Use only the provided context plus conversation. If data is missing, supply reasonable placeholders.
User question:
{user_query}
""".strip()


def _invoke_bedrock(payload: Dict[str, Any]) -> Dict[str, Any]:
    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        accept="application/json",
        contentType="application/json",
        body=json.dumps(payload),
    )
    body = response.get("body")
    body_bytes = body.read() if hasattr(body, "read") else body
    if isinstance(body_bytes, (bytes, bytearray)):
        body_bytes = body_bytes.decode("utf-8")
    if not body_bytes:
        raise ValueError("Empty Bedrock body")
    return json.loads(body_bytes)


def _collect_text(response_json: Dict[str, Any]) -> str:
    content_blocks = response_json.get("content") or []
    chunks: List[str] = []
    for block in content_blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            chunks.append(block.get("text") or "")
    return "".join(chunks).strip()


def _format_history(messages: List[Dict[str, Any]]) -> str:
    if not messages:
        return "No prior conversation."
    trimmed = messages[-MAX_HISTORY_TURNS:]
    lines: List[str] = []
    for turn in trimmed:
        role = turn.get("role", "user").upper()
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{role}: {content[:600]}")
    snippet = "\n".join(lines).strip()
    return snippet[:2000] if snippet else "No usable history."


def _time_window_guidance(time_hours: int) -> str:
    if time_hours <= 24:
        return "Interview within 24h - focus on calm confidence and rapid refreshers."
    if time_hours <= 48:
        return "Interview within 48h - balance targeted drills with rest."
    return "Interview more than 48h away - build a structured multi-day plan."


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _default_response(sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "answer_markdown": "I couldn't gather reliable interview intel right now. Please try again shortly.",
        "sections": [],
        "sources": sources,
    }
