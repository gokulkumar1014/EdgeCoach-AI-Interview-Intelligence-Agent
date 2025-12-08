"""
EdgeCoach AI - Tavily retrieval

This module queries Tavily and fetches external URLs for interview intel. It:
- Sends multiple Tavily search queries for company/role interview content.
- Fetches pages (HTML/PDF), cleans text, and trims to safe length.
- Returns deduplicated sources with id, title, snippet, domain, and content.
"""
from typing import List, Dict, Any, Optional
import io
import logging
import os
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

try:
    import trafilatura  # type: ignore
except ImportError:  # pragma: no cover
    trafilatura = None

try:
    from PyPDF2 import PdfReader  # type: ignore
except ImportError:  # pragma: no cover
    PdfReader = None

try:
    from bs4 import BeautifulSoup  # type: ignore
except ImportError:  # pragma: no cover
    BeautifulSoup = None


TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_URL = "https://api.tavily.com/search"
_MAX_CONTENT_CHARS = 10_000
_MIN_CONTENT_CHARS = 300
_REQUEST_TIMEOUT = 15

logger = logging.getLogger(__name__)


def fetch_interview_sources(
    company: str,
    role: str,
    max_sources: int = 3,
) -> List[Dict[str, Any]]:
    """
    Retrieve up to 3 high-quality interview sources that include URL, title, snippet, domain, and cleaned content.
    """
    if not TAVILY_API_KEY:
        logger.warning("TAVILY_API_KEY missing; cannot fetch sources.")
        return []

    max_sources = max(1, min(max_sources or 1, 3))

    company = (company or "").strip()
    role = (role or "").strip()
    if not company and not role:
        logger.warning("Neither company nor role provided; skipping Tavily call.")
        return []

    queries = [
        f"{company} {role} interview experience".strip(),
        f"{company} {role} interview process".strip(),
        f"glassdoor {company} interview questions".strip(),
        f"reddit {company} interview questions".strip(),
        f"interview tips for {company}".strip(),
    ]

    raw_candidates: List[Dict[str, Any]] = []
    seen_urls = set()

    for query in queries:
        if len(raw_candidates) >= max_sources:
            break
        payload = {
            "api_key": TAVILY_API_KEY,
            "query": query,
            "search_depth": "advanced",
            "include_raw_content": True,
            "max_results": max_sources,
        }
        try:
            response = requests.post(
                TAVILY_URL,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as exc:
            logger.warning("Tavily search failed for query '%s': %s", query, exc)
            continue

        for item in data.get("results", []):
            if len(raw_candidates) >= max_sources:
                break

            url = (item.get("url") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            fallback_title = f"{company} {role}".strip() or "Interview source"
            title = (item.get("title") or url or fallback_title).strip()
            snippet = (
                item.get("snippet")
                or item.get("content")
                or item.get("answer")
                or title
                or ""
            ).strip()

            raw_candidates.append(
                {
                    "url": url,
                    "title": title,
                    "snippet": snippet,
                    "source": _derive_source(url),
                    "raw_content": item.get("raw_content") or "",
                    "content": item.get("content") or "",
                }
            )

    if not raw_candidates:
        return []

    enriched_sources: List[Dict[str, Any]] = []
    max_workers = min(4, len(raw_candidates))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_fetch_url_content, candidate): candidate for candidate in raw_candidates
        }
        for future in as_completed(future_map):
            candidate = future_map[future]
            content = ""
            try:
                content = future.result()
            except Exception as exc:  # pragma: no cover
                logger.warning("Content fetching crashed for %s: %s", candidate.get("url"), exc)

            if not content:
                continue

            enriched_sources.append(
                {
                    "id": f"S{len(enriched_sources) + 1}",
                    "url": candidate["url"],
                    "title": candidate["title"],
                    "source": candidate["source"],
                    "snippet": candidate["snippet"],
                    "content": content[:_MAX_CONTENT_CHARS],
                }
            )

            if len(enriched_sources) >= max_sources:
                break

    return enriched_sources


def _fetch_url_content(candidate: Dict[str, Any]) -> str:
    """
    Fetch and extract high-quality text from a URL with multiple fallbacks.
    """
    url = candidate.get("url", "")
    if not url:
        return ""

    fallback_texts = [
        candidate.get("raw_content", ""),
        candidate.get("content", ""),
        candidate.get("snippet", ""),
    ]

    cleaned_text = ""
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "InterviewIntelAgent/1.0"},
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        content_type = (response.headers.get("Content-Type") or "").lower()

        if "pdf" in content_type or url.lower().endswith(".pdf"):
            cleaned_text = _load_pdf(response.content)
        else:
            page_text = response.text
            cleaned_text = _extract_via_trafilatura(page_text, url)
            if not cleaned_text:
                cleaned_text = _extract_via_bs4(page_text)

    except requests.RequestException as exc:
        logger.info("HTTP fetch failed for %s: %s", url, exc)

    if not cleaned_text:
        for fallback in fallback_texts:
            cleaned_fallback = _clean_text(fallback)
            if len(cleaned_fallback) >= _MIN_CONTENT_CHARS:
                cleaned_text = cleaned_fallback
                break

    cleaned_text = _clean_text(cleaned_text)
    if len(cleaned_text) < _MIN_CONTENT_CHARS:
        return ""

    return cleaned_text[:_MAX_CONTENT_CHARS]


def _load_pdf(content: bytes) -> str:
    if not PdfReader:
        logger.info("PyPDF2 not installed; skipping PDF extraction.")
        return ""
    try:
        reader = PdfReader(io.BytesIO(content))  # type: ignore
    except Exception as exc:  # pragma: no cover
        logger.info("Failed to load PDF: %s", exc)
        return ""

    text_chunks: List[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:  # pragma: no cover
            text = ""
        text_chunks.append(text)
    return _clean_text("\n".join(text_chunks))


def _extract_via_trafilatura(page_text: str, url: str) -> str:
    if not trafilatura:
        return ""
    try:
        return trafilatura.extract(page_text, url=url, include_comments=False) or ""
    except Exception as exc:  # pragma: no cover
        logger.info("Trafilatura extraction failed for %s: %s", url, exc)
        return ""


def _extract_via_bs4(page_text: str) -> str:
    if not BeautifulSoup:
        return ""
    try:
        soup = BeautifulSoup(page_text, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except Exception as exc:  # pragma: no cover
        logger.info("BeautifulSoup extraction failed: %s", exc)
        return ""


def _derive_source(url: str) -> str:
    try:
        parsed = urlparse(url)
        hostname = parsed.netloc.lower()
        if hostname.startswith("www."):
            hostname = hostname[4:]
        return hostname or "web"
    except Exception:
        return "web"


def _clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    stripped = "".join(ch if ch.isprintable() else " " for ch in str(text))
    collapsed = []
    last_space = False
    for char in stripped:
        if char.isspace():
            if not last_space:
                collapsed.append(" ")
            last_space = True
        else:
            collapsed.append(char)
            last_space = False
    return "".join(collapsed).strip()
