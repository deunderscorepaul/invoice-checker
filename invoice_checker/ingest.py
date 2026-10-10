"""Text-layer PDF extraction and OCR fallback.

Extraction order: try the PDF text layer first, and only fall back to OCR when a document has no
usable text layer. OCR'ing a digital PDF is a classic self-inflicted accuracy loss -- it adds
recognition errors to text that was already perfect -- so the pipeline will not do it.
"""
from __future__ import annotations

import re
import zlib
from pathlib import Path

# Tokeniser for the text-showing subset of the PDF content-stream syntax.
#
# ``T*`` must NOT be wrapped in ``\b`` assertions: ``*`` is a non-word character, so ``\bT\*\b``
# can never match between "*" and the following whitespace. That mistake made the tokeniser skip
# every T* line break and silently concatenate whole documents into one line.
TOKEN_RE = re.compile(
    rb"\((?:\\.|[^\\()])*\)"      # literal string
    rb"|<[0-9A-Fa-f\s]*>"          # hex string
    rb"|T\*"                       # next line (the T* operator)
    rb"|\bTJ\b|\bTj\b"             # show text
    rb"|\bTd\b|\bTD\b|\bTm\b"      # position text
    rb"|\bBT\b|\bET\b"             # begin / end text object
)

UNESCAPE = {b"n": b"\n", b"r": b"\r", b"t": b"\t", b"b": b"\b", b"f": b"\f",
            b"(": b"(", b")": b")", b"\\": b"\\"}


class IngestError(RuntimeError):
    pass


def _decode_literal(raw: bytes) -> bytes:
    """Undo PDF literal-string escapes."""
    out = bytearray()
    index = 0
    while index < len(raw):
        char = raw[index:index + 1]
        if char == b"\\" and index + 1 < len(raw):
            nxt = raw[index + 1:index + 2]
            if nxt.isdigit():
                octal = raw[index + 1:index + 4]
                match = re.match(rb"[0-7]{1,3}", octal)
                out.append(int(match.group(0), 8) & 0xFF)
                index += 1 + len(match.group(0))
                continue
            out.extend(UNESCAPE.get(nxt, nxt))
            index += 2
            continue
        out.append(raw[index])
        index += 1
    return bytes(out)


def _decode_hex(raw: bytes) -> bytes:
    cleaned = re.sub(rb"\s+", b"", raw)
    if len(cleaned) % 2:
        cleaned += b"0"
    try:
        return bytes.fromhex(cleaned.decode("ascii"))
    except ValueError:
        return b""


def extract_text_from_pdf_bytes(data: bytes) -> str:
    """Extract text from uncompressed or Flate-compressed content streams.

    Handles the common case (one or more content streams, Tj/TJ text operators) and returns an
    empty string rather than raising when a document uses none of them -- the caller then decides
    whether to OCR.
    """
    pieces: list[str] = []
    for match in re.finditer(rb"stream\r?\n", data):
        start = match.end()
        end = data.find(b"endstream", start)
        if end == -1:
            continue
        payload = data[start:end]
        if payload.endswith(b"\r\n"):
            payload = payload[:-2]
        elif payload.endswith(b"\n"):
            payload = payload[:-1]
        stream = payload
        if b"FlateDecode" in data[max(0, match.start() - 400):match.start()]:
            try:
                stream = zlib.decompress(payload)
            except zlib.error:
                try:
                    stream = zlib.decompressobj().decompress(payload)
                except zlib.error:
                    continue
        if b"Tj" not in stream and b"TJ" not in stream:
            continue
        pieces.append(_interpret_content_stream(stream))
    return "\n".join(part for part in pieces if part.strip())


def _interpret_content_stream(stream: bytes) -> str:
    """Walk text operators and reconstruct lines.

    Ordering matters and is easy to get wrong. A text-showing operator emits the *current* text, so
    ``pending`` must be flushed when it is encountered -- not when the *following* positioning
    operator arrives. An earlier version appended the newline before flushing, which silently
    concatenated every line of the document into one string and destroyed the regex extractor.
    """
    out: list[str] = []
    pending: list[str] = []

    def flush() -> None:
        if pending:
            out.append("".join(pending))
            pending.clear()

    for token in TOKEN_RE.finditer(stream):
        raw = token.group(0)
        if raw in (b"Tj", b"TJ"):
            # Both show the accumulated text; TJ additionally carries its own micro-spacing which we
            # deliberately ignore rather than trying to reconstruct kerning.
            flush()
            continue
        if raw in (b"T*", b"Td", b"TD", b"Tm"):
            flush()
            if out and out[-1] != "\n":
                out.append("\n")
            continue
        if raw in (b"BT", b"ET"):
            continue
        if raw.startswith(b"("):
            pending.append(_decode_literal(raw[1:-1]).decode("cp1252", errors="replace"))
        elif raw.startswith(b"<"):
            pending.append(_decode_hex(raw[1:-1]).decode("utf-16-be", errors="replace"))
    flush()

    text = "".join(out)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_pdf(path) -> str:
    path = Path(path)
    data = path.read_bytes()
    try:
        import pypdf  # type: ignore

        reader = pypdf.PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if text.strip():
            return text.strip()
    except Exception:
        # Fall through to the built-in extractor; pypdf is a bonus, not a requirement.
        pass
    return extract_text_from_pdf_bytes(data)


def ocr_image(path) -> str:
    """OCR via pytesseract if available. Raises IngestError when OCR is not installed."""
    try:
        import pytesseract  # type: ignore
        from PIL import Image
    except ImportError as exc:
        raise IngestError(
            "OCR requested but pytesseract/Pillow is not installed; "
            "install pytesseract and the Tesseract binary, or supply a text-layer PDF"
        ) from exc
    try:
        with Image.open(path) as image:
            return pytesseract.image_to_string(image, lang="deu+eng").strip()
    except Exception as exc:
        raise IngestError(f"OCR failed for {path}: {exc}") from exc


def ingest_document(path, *, allow_ocr: bool = True) -> tuple[str, str]:
    """Return ``(text, method)`` where method is one of text/ocr/none."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = extract_text_from_pdf(path)
        if text.strip():
            return text, "text"
        if allow_ocr:
            return ocr_image(path), "ocr"
        return "", "none"
    if suffix == ".txt":
        return path.read_text(encoding="utf-8"), "text"
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        if not allow_ocr:
            return "", "none"
        return ocr_image(path), "ocr"
    raise IngestError(f"unsupported document type: {suffix or path.name}")
