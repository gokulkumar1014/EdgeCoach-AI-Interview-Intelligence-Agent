"""
EdgeCoach AI - Lambda handler

This module defines the AWS Lambda entrypoint for the interview
intelligence assistant. It:
- Parses the incoming HTTP event from the Function URL.
- Restores cached agent state from prior turns.
- Calls the intent classifier, retrieval module, and analysis engine.
- Routes general Q&A queries to a lightweight Bedrock call.
"""

import json
from typing import Any, Dict, List, Tuple

import boto3

from bedrock_intent import extract_intent as detect_intent
from tavily_retrieval import fetch_interview_sources
from analysis_engine import synthesize_interview_answer

STATE_PREFIX = "__agent_state__:"
bedrock = boto3.client("bedrock-runtime")


def _generate_general_answer(user_query: str, messages: List[Dict[str, Any]]) -> str:
    """
    Use Claude (Bedrock) to answer general-purpose queries when the user
    is NOT asking about interview preparation or interview intel.
    """
    try:
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 500,
            "temperature": 0.5,
            "system": "You are a helpful and friendly general-purpose AI assistant. Answer clearly and concisely.",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": user_query}]}
            ],
        }

        response = bedrock.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            accept="application/json",
            contentType="application/json",
            body=json.dumps(payload),
        )

        body = response.get("body")
        body_bytes = body.read() if hasattr(body, "read") else body
        if isinstance(body_bytes, (bytes, bytearray)):
            body_bytes = body_bytes.decode("utf-8")
        response_json = json.loads(body_bytes or "{}")

        chunks = [
            block.get("text", "")
            for block in response_json.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "".join(chunks).strip() or "I'm here! How can I help you?"
    except Exception:
        return "I'm here! How can I help you?"


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    try:
        body = _parse_event_body(event)
        user_query = (body.get("query") or "").strip()
        raw_messages = body.get("messages") or []

        if not user_query:
            return _error_response(400, "Missing 'query' in request body.")

        if not isinstance(raw_messages, list):
            return _error_response(400, "'messages' must be a list of turns.")

        # Conversation history without agent-state system messages,
        # and any cached state that was stored previously.
        conversation_history, cached_state = _extract_agent_state(raw_messages)

        # --- Intent detection from Bedrock ---
        latest_intent = detect_intent(
            user_query,
            history=conversation_history,
        )

        # --- OVERRIDE: treat pure concept questions as general chat ---
        q_lower = user_query.lower().strip()
        concept_prefixes = (
            "what is ",
            "what are ",
            "explain ",
            "define ",
            "how does ",
            "how do ",
            "describe ",
            "difference between ",
            "compare ",
            "when is ",
            "why is ",
        )
        is_concept_question = (
            any(q_lower.startswith(p) for p in concept_prefixes)
            and "interview" not in q_lower
            and "round" not in q_lower
        )

        if not isinstance(latest_intent, dict):
            latest_intent = {}

        if is_concept_question:
            # Force this turn to be handled as general Q&A,
            # not as interview intel, and clear any stale interview state.
            latest_intent["wants_interview_intel"] = False
            latest_intent["company"] = ""
            latest_intent["role"] = ""

            cached_state = {
                "company": "",
                "role": "",
                "sources": [],
            }

        # Append the current user message for downstream steps
        history_with_user = conversation_history + [
            {"role": "user", "content": user_query}
        ]

        # === GENERAL CHAT MODE ROUTING ===
        if not latest_intent.get("wants_interview_intel", False):
            general_answer = _generate_general_answer(user_query, history_with_user)
            updated_history = history_with_user + [
                {"role": "assistant", "content": general_answer}
            ]

            response_payload = {
                "intent": latest_intent,
                "answer": general_answer,
                "sources": [],
                "messages": updated_history,
            }

            return {
                "statusCode": 200,
                "body": json.dumps(response_payload),
            }

        # === INTERVIEW-INTEL FLOW (with Tavily) ===
        company = (latest_intent.get("company") or "").strip()
        role = (latest_intent.get("role") or "").strip()
        
        if latest_intent.get("wants_interview_intel") and not company and cached_state.get("company"):
            company = cached_state["company"]
            if not role:
                 role = cached_state.get("role", "")

        sources = cached_state.get("sources") or []
        previous_company = cached_state.get("company", "")
        previous_role = cached_state.get("role", "")

        should_refresh_sources = (
            not sources
            or company != previous_company
            or role != previous_role
        )

        if latest_intent.get("wants_interview_intel", True) and should_refresh_sources:
            refreshed_sources = fetch_interview_sources(
                company,
                role,
                max_sources=5,
            )
            if refreshed_sources:
                sources = refreshed_sources
            elif not sources:
                sources = []

        answer_payload = synthesize_interview_answer(
            user_query=user_query,
            intent=latest_intent,
            sources=sources,
            messages=history_with_user,
        )

        answer_markdown = answer_payload.get("answer_markdown", "")
        updated_history = history_with_user + [
            {"role": "assistant", "content": answer_markdown}
        ]

        agent_state_message = _build_agent_state_message(
            company=company,
            role=role,
            sources=sources,
        )
        updated_history.append(agent_state_message)

        response_payload = {
            "intent": latest_intent,
            "answer": answer_markdown,
            "sources": answer_payload.get("sources", []),
            "messages": updated_history,
        }

        return {
            "statusCode": 200,
            "body": json.dumps(response_payload),
        }

    except Exception as exc:
        return _error_response(500, f"Unexpected server error: {exc}")


def _parse_event_body(event: Dict[str, Any]) -> Dict[str, Any]:
    body_value = event.get("body") or {}
    if isinstance(body_value, str):
        try:
            return json.loads(body_value)
        except json.JSONDecodeError:
            return {}
    if isinstance(body_value, dict):
        return body_value
    return {}


def _extract_agent_state(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cleaned_messages: List[Dict[str, Any]] = []
    agent_state: Dict[str, Any] = {
        "company": "",
        "role": "",
        "sources": [],
    }

    for message in messages:
        if (
            isinstance(message, dict)
            and message.get("role") == "system"
            and isinstance(message.get("content"), str)
            and message["content"].startswith(STATE_PREFIX)
        ):
            state_content = message["content"][len(STATE_PREFIX):]
            try:
                data = json.loads(state_content)
                if isinstance(data, dict):
                    agent_state.update(
                        {
                            "company": data.get("company", ""),
                            "role": data.get("role", ""),
                            "sources": data.get("sources", []),
                        }
                    )
            except json.JSONDecodeError:
                continue
        else:
            cleaned_messages.append(message)

    return cleaned_messages, agent_state


def _build_agent_state_message(company: str, role: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    payload = {
        "company": company,
        "role": role,
        "sources": sources,
    }
    return {
        "role": "system",
        "content": f"{STATE_PREFIX}{json.dumps(payload)}",
    }


def _error_response(status: int, message: str) -> Dict[str, Any]:
    return {
        "statusCode": status,
        "body": json.dumps({"error": message}),
    }
