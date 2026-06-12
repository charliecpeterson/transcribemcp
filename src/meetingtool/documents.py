"""Document extraction + chunking.

Text extraction per format (txt/md/pdf/docx) and paragraph-aware chunking.
Chunks target ~CHUNK_TARGET chars, respecting paragraph boundaries when
possible, splitting on sentences or hard-cutting only as a fallback. Chunks
are the unit of retrieval via FTS5 — small enough to return as snippets,
large enough that bm25 sees meaningful context.
"""
from __future__ import annotations

import re
from pathlib import Path

CHUNK_TARGET = 800
CHUNK_MAX = 1200


def extract_text(path: Path) -> tuple[str, str]:
    """Return (text, kind) for a supported file. Raises on unsupported types."""
    suffix = path.suffix.lower().lstrip(".")
    if suffix in ("txt", "md", "markdown"):
        return path.read_text(encoding="utf-8", errors="replace"), "md" if suffix == "markdown" else suffix
    if suffix == "pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages), "pdf"
    if suffix == "docx":
        import docx

        doc = docx.Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs]
        return "\n\n".join(paragraphs), "docx"
    raise ValueError(
        f"unsupported document type: .{suffix} "
        "(supported: .txt, .md, .markdown, .pdf, .docx)"
    )


_PARA_SPLIT = re.compile(r"\n\s*\n+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def chunk_text(text: str, target: int = CHUNK_TARGET, max_size: int = CHUNK_MAX) -> list[str]:
    """Split text into chunks near `target` chars, never exceeding `max_size`.

    Preserves paragraph boundaries first; falls back to sentence splits for
    oversized paragraphs, then hard character slicing if a single sentence is
    still too long.
    """
    paragraphs = [p.strip() for p in _PARA_SPLIT.split(text) if p.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_len
        if buf:
            chunks.append("\n\n".join(buf))
            buf = []
            buf_len = 0

    for para in paragraphs:
        if len(para) > max_size:
            flush()
            for piece in _split_oversized(para, target, max_size):
                chunks.append(piece)
            continue
        added = len(para) + (2 if buf else 0)
        if buf_len + added > target and buf:
            flush()
        buf.append(para)
        buf_len += added
    flush()
    return chunks


def _split_oversized(para: str, target: int, max_size: int) -> list[str]:
    sentences = _SENT_SPLIT.split(para)
    out: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for sent in sentences:
        if len(sent) > max_size:
            if buf:
                out.append(" ".join(buf))
                buf, buf_len = [], 0
            for i in range(0, len(sent), max_size):
                out.append(sent[i : i + max_size])
            continue
        added = len(sent) + (1 if buf else 0)
        if buf_len + added > target and buf:
            out.append(" ".join(buf))
            buf, buf_len = [], 0
        buf.append(sent)
        buf_len += added
    if buf:
        out.append(" ".join(buf))
    return out
