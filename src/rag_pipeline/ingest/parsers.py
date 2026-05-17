from __future__ import annotations

from pathlib import Path

SUPPORTED_EXTS = {".pdf", ".md", ".markdown", ".txt", ".html", ".htm"}


def source_type_of(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return "pdf"
    if ext in {".md", ".markdown"}:
        return "md"
    if ext == ".txt":
        return "txt"
    if ext in {".html", ".htm"}:
        return "html"
    raise ValueError(f"unsupported extension: {ext}")


def parse_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    return "\n\n".join(parts).strip()


def parse_html(path: Path) -> str:
    from bs4 import BeautifulSoup

    with open(path, encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def parse_text(path: Path) -> str:
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read().strip()


def parse(path: Path) -> tuple[str, str]:
    """Return (source_type, text)."""
    stype = source_type_of(path)
    if stype == "pdf":
        return stype, parse_pdf(path)
    if stype == "html":
        return stype, parse_html(path)
    return stype, parse_text(path)
