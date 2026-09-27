"""Resolve explicit Hitomi IDs, titles, and URLs into safe crawl targets."""

import re
import unicodedata
from urllib.parse import quote, urlparse


def search_url_for_title(title: str) -> str:
    title = title.strip()
    if not title:
        raise ValueError("Title must not be empty")
    # The site's search parser treats ':' as a namespace operator and a leading
    # '-' as an exclusion. Search broader words, then match the actual title.
    terms = [part.lstrip("-–—") for part in re.sub(r"[:：]", " ", title).split()]
    query = " ".join(part for part in terms if part and part.casefold() != "or")
    if not query:
        raise ValueError("Title has no searchable terms")
    return "https://hitomi.la/search.html?" + quote(query, safe="")


def normalize_title(title: str) -> str:
    text = unicodedata.normalize("NFKC", title).casefold()
    text = text.translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'}))
    return " ".join(text.split())


def title_matches(candidate: str, requested: str, mode: str = "exact") -> bool:
    candidate_key = normalize_title(candidate)
    requested_key = normalize_title(requested)
    if mode == "exact":
        return candidate_key == requested_key
    if mode == "contains":
        return requested_key in candidate_key
    raise ValueError(f"Unsupported title match mode: {mode}")


def canonical_gallery_type(value: str, allowed: list[str]) -> str | None:
    """Return the configured type spelling even when the site changes case."""
    return {kind.casefold(): kind for kind in allowed}.get(value.strip().casefold())


def parse_hitomi_url(url: str) -> tuple[str, str]:
    """Return ('gallery', ID) or ('listing', URL); reject foreign URLs."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "hitomi.la":
        raise ValueError(f"Expected an https://hitomi.la URL: {url}")
    direct = re.fullmatch(r"/(?:reader|galleries)/(\d+)\.html", parsed.path)
    if direct:
        return "gallery", direct.group(1)
    gallery = re.fullmatch(r"/[^/]+/[^/]+-(\d+)\.html", parsed.path)
    if gallery:
        return "gallery", gallery.group(1)
    if parsed.path == "/" or parsed.path.endswith(".html"):
        return "listing", url
    raise ValueError(f"Unsupported Hitomi URL: {url}")


def parse_gg_routing(script: str) -> tuple[str, int, dict[int, int]]:
    """Read the site's current webp host rule, including either default value."""
    prefix_match = re.search(r"\bb\s*:\s*['\"]([^'\"]+/)['\"]", script)
    function_match = re.search(
        r"\bm\s*:\s*function\s*\(\s*g\s*\)\s*\{(.*?)return\s+o\s*;",
        script, re.DOTALL,
    )
    if not prefix_match or not function_match:
        raise ValueError("Unexpected image routing script")
    body = function_match.group(1)
    default_match = re.search(r"\bvar\s+o\s*=\s*([01])\s*;", body)
    switch_match = re.search(r"switch\s*\(\s*g\s*\)\s*\{(.*?)\}", body, re.DOTALL)
    if not default_match or not switch_match:
        raise ValueError("Unexpected image routing switch")
    default = int(default_match.group(1))
    overrides = {}
    groups = re.findall(
        r"((?:\s*case\s+\d+\s*:\s*)+)o\s*=\s*([01])\s*;\s*break\s*;",
        switch_match.group(1), re.DOTALL,
    )
    if not groups:
        raise ValueError("No image routing cases found")
    for cases, value in groups:
        for number in re.findall(r"case\s+(\d+)\s*:", cases):
            overrides[int(number)] = int(value)
    return prefix_match.group(1), default, overrides
