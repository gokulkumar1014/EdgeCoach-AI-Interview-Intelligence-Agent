"""
EdgeCoach AI - Streamlit frontend

This module renders the chat UI and talks to the Lambda backend. It:
- Sends user queries plus history to the Function URL.
- Normalizes Lambda responses and preserves backend messages.
- Renders chat bubbles with optional web source expanders.
"""
import json
from html import escape
from typing import Any, Dict, List

import requests
import streamlit as st

# ==========================
# CONFIG
# ==========================

API_URL = "https://wmppa4oufkcu4bx6xv2g77uoge0goofg.lambda-url.us-east-1.on.aws/"


# ==========================
# HELPERS
# ==========================

def md_to_html(value: str) -> str:
    """
    Very small markdown-to-HTML helper.
    Streamlit renders markdown directly, so this is only used
    for minimal escaping (if needed later).
    """
    safe = escape(value or "")
    return safe.replace("\n", "<br>")


def normalize_lambda_payload(raw: Any) -> Dict[str, Any]:
    """
    Normalize the Lambda HTTP response body into a dict with:
      { intent, answer, sources, messages }
    Handles both:
      - direct JSON
      - { "statusCode": 200, "body": "{...}" }
    """
    data = raw

    # If we already have a dict and it has "body", unwrap
    if isinstance(data, dict) and "body" in data and len(data.keys()) <= 3:
        body_val = data.get("body")
        if isinstance(body_val, str):
            try:
                data = json.loads(body_val)
            except json.JSONDecodeError as exc:
                raise ValueError("Lambda returned invalid JSON in 'body'.") from exc
        elif isinstance(body_val, dict):
            data = body_val

    if not isinstance(data, dict):
        raise ValueError("Lambda response must be a JSON object.")

    # Basic shape validation
    if "answer" not in data:
        raise ValueError("Lambda response missing 'answer' field.")

    return data


def call_backend(query: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Call the Lambda backend with the current query and full message history.
    """
    payload = {"query": query, "messages": messages}
    resp = requests.post(API_URL, json=payload, timeout=60)
    resp.raise_for_status()

    raw_json: Any = resp.json()
    normalized = normalize_lambda_payload(raw_json)
    return normalized


def render_sources_block(sources: List[Dict[str, Any]]) -> None:
    """
    Render 'Web Sources Used' cards under the assistant message.
    Only called when there are actually some sources.
    """
    if not sources:
        return

    st.markdown(
        "<div style='margin-top:0.75rem; font-size:0.7rem; "
        "text-transform:uppercase; letter-spacing:0.16em; "
        "color:#7b87b9;'>Web Sources Used</div>",
        unsafe_allow_html=True,
    )

    for idx, src in enumerate(sources, start=1):
        url = str(src.get("url") or "").strip()
        title = str(src.get("title") or url or "Untitled").strip()
        domain = str(src.get("source") or "").strip() or "web"
        snippet = str(src.get("snippet") or "").strip()

        label = f"Source {idx} - {domain}"

        with st.expander(label, expanded=False):
            # Title with link
            if url:
                st.markdown(f"**[{title}]({url})**")
            else:
                st.markdown(f"**{title}**")

            if snippet:
                st.markdown(f"\n{snippet}")


def strip_system_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    From the backend messages, keep only user/assistant for UI.
    """
    ui_messages: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role in ("user", "assistant") and content:
            ui_messages.append({"role": role, "content": content})
    return ui_messages


# Avatars for chat roles
ROLE_AVATARS = {
    "assistant": "💡",
    "user": "❤️",
}


# ==========================
# STREAMLIT PAGE SETUP
# ==========================

st.set_page_config(
    page_title="EdgeCoach AI",
    page_icon=":speech_balloon:",
    layout="wide",
)

CUSTOM_CSS = """
<style>
#MainMenu, header, footer {visibility: hidden;}

.stApp {
    background: radial-gradient(120% 120% at 20% 20%, #f8fbff 0%, #eef2ff 40%, #e5ebff 100%);
    color: #0f1b40;
    font-family: "Inter", "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont,
                 "Helvetica Neue", sans-serif;
}

/* Center content and give it a clean max width */
.block-container {
    max-width: 1100px;
    padding-top: 1.25rem;
    padding-bottom: 2.5rem;
}

/* Hero card */
.hero {
    background: linear-gradient(135deg, rgba(79, 110, 247, 0.12), rgba(24, 44, 120, 0.08));
    border: 1px solid rgba(79, 110, 247, 0.25);
    border-radius: 18px;
    padding: 1.25rem 1.4rem;
    box-shadow: 0 12px 32px rgba(17, 33, 77, 0.12);
}
.hero h1 {
    margin: 0 0 0.35rem 0;
    color: #0f1b40;
}
.hero p {
    margin: 0.1rem 0;
    line-height: 1.5;
    color: #223056;
}

.hero-divider {
    margin-top: 1rem;
    margin-bottom: 1.4rem;
    border: none;
    height: 1px;
    background: linear-gradient(90deg, rgba(79,110,247,0.6), rgba(101,124,255,0.05));
}

/* Chat input styling */
[data-testid="stChatInput"] {
    max-width: 1000px;
    margin: 0 auto;
}
[data-testid="stChatInput"] textarea {
    border-radius: 999px;
    padding: 0.95rem 1.25rem;
    border: 1px solid #c8d5ff;
    background: rgba(255,255,255,0.9);
    color: #0f1b40;
    box-shadow: 0 10px 28px rgba(22, 45, 120, 0.16);
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #4b5563;
}
[data-testid="stChatInput"] button {
    border-radius: 999px;
    padding: 0.45rem 1rem;
    background: linear-gradient(135deg, #4458f7, #6f9bff);
    border: none;
    color: #ffffff;
    font-weight: 700;
    box-shadow: 0 14px 26px rgba(79, 110, 247, 0.35);
}

/* Make chat column centered */
.stChatMessage {
    max-width: 920px;
    margin-left: auto;
    margin-right: auto;
}

/* Assistant bubble tweaks */
.stChatMessage[data-testid="stChatMessage-role-assistant"] > div {
    background: rgba(255, 255, 255, 0.92);
    border-radius: 18px;
    border: 1px solid #d9e2ff;
    box-shadow: 0 12px 34px rgba(31, 59, 119, 0.12);
    color: #0f1b40;
}

/* Remove Streamlit's default user bubble and apply our own */
.stChatMessage[data-testid="stChatMessage-role-user"] > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
.stChatMessage[data-testid="stChatMessage-role-user"] .stChatMessageContent,
.stChatMessage[data-testid="stChatMessage-role-user"] [data-testid="stMarkdownContainer"],
.stChatMessage[data-testid="stChatMessage-role-user"] [data-testid="stMarkdownContainer"] > div,
.stChatMessage[data-testid="stChatMessage-role-user"] .stMarkdown,
.stChatMessage[data-testid="stChatMessage-role-user"] .stMarkdown > div {
    background: linear-gradient(135deg, #ff5757, #ff7676) !important;
    color: #ffffff !important;
    border-radius: 18px !important;
    padding: 1rem 1.2rem !important;
    border: 1px solid #ff8a8a !important;
    box-shadow: 0 14px 28px rgba(255,87,87,0.35) !important;
    font-weight: 600 !important;
}
.stChatMessage[data-testid="stChatMessage-role-user"] .stMarkdown p {
    margin: 0 !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    "<div class='hero'>"
    "<h1 style='font-size:2rem; font-weight:700; color:#0f1b40; margin-bottom:0.35rem;'>EdgeCoach AI</h1>"
    "<p style='color:#223056;'>Learn clearly. Prepare confidently. Perform exceptionally.</p>"
    "<p style='color:#223056;'>For upcoming interviews, I gather real-world candidate experiences from the web and distill them into company-specific insights and prep plans -- automatically.</p>"
    "</div>",
    unsafe_allow_html=True,
)
st.markdown("<hr class='hero-divider'/>", unsafe_allow_html=True)

# ==========================
# SESSION STATE
# ==========================

# Messages used ONLY for UI (user + assistant)
if "ui_messages" not in st.session_state:
    st.session_state["ui_messages"] = []

# Full messages (including system state) sent to backend
if "backend_messages" not in st.session_state:
    st.session_state["backend_messages"] = []


# ==========================
# RENDER EXISTING CHAT
# ==========================

if not st.session_state["ui_messages"]:
    with st.chat_message("assistant", avatar=ROLE_AVATARS.get("assistant")):
        st.markdown(
            "Hi, I'm EdgeCoach AI - your personal interview coach.\n\n"
            "- Ask me anything: concepts, coding, ML/AI, or general questions.\n"
            "- Mention an interview and I'll pull real-world candidate experiences, surface web sources, and craft a tailored prep plan."
        )
else:
    for msg in st.session_state["ui_messages"]:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        sources = msg.get("sources") or []

        with st.chat_message(role, avatar=ROLE_AVATARS.get(role)):
            st.markdown(content)

            # Only assistants can have sources
            if role == "assistant" and sources:
                render_sources_block(sources)


# ==========================
# CHAT INPUT & BACKEND CALL
# ==========================

prompt = st.chat_input("Send a message")

if prompt is not None:
    user_text = prompt.strip()
    if not user_text:
        st.warning("Please type something before sending.")
    else:
        # 1) Append user message to UI state
        st.session_state["ui_messages"].append(
            {"role": "user", "content": user_text}
        )

        # 2) Call backend with full history (excluding this turn, query is sent separately)
        try:
            history_for_backend = st.session_state["backend_messages"]
            response = call_backend(user_text, history_for_backend)
        except requests.exceptions.RequestException:
            # Network error -> append an assistant error message
            st.session_state["ui_messages"].append(
                {
                    "role": "assistant",
                    "content": (
                        "I couldn't reach the backend just now. "
                        "Please try again in a moment."
                    ),
                }
            )
        except ValueError:
            # Parsing / shape error -> append an assistant error message
            st.session_state["ui_messages"].append(
                {
                    "role": "assistant",
                    "content": (
                        "I received an unexpected response from the backend. "
                        "Let's try again."
                    ),
                }
            )
        else:
            # 3) Update backend history with what Lambda returns
            backend_messages = response.get("messages")
            if isinstance(backend_messages, list):
                st.session_state["backend_messages"] = backend_messages

            intent = response.get("intent") or {}
            if not isinstance(intent, dict):
                intent = {}

            answer = str(response.get("answer") or "").strip()
            if not answer:
                answer = "I don't have an answer for that yet."

            all_sources = response.get("sources")
            source_list: List[Dict[str, Any]] = (
                all_sources if isinstance(all_sources, list) else []
            )

            show_sources = bool(intent.get("wants_interview_intel")) and bool(
                source_list
            )

            ui_assistant_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": answer,
            }
            if show_sources:
                ui_assistant_msg["sources"] = source_list

            # Append assistant message to UI state
            st.session_state["ui_messages"].append(ui_assistant_msg)

        # 4) Force a rerun so the top loop re-renders the full chat cleanly
        st.rerun()
