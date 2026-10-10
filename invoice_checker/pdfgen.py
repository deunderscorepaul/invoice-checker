"""Rechnungen rendern: Text, Scan-Optik und ein minimaler PDF-Schreiber.

Der PDF-Schreiber sind ~150 Zeilen Standardbibliothek, damit niemand reportlab
installieren muss. Erzeugt einseitige PDFs mit Flate-komprimiertem Content-Stream
und Helvetica (WinAnsi) -- genug, um den echten Textebene-Pfad zu testen.
"""
from __future__ import annotations

import zlib
from pathlib import Path
from typing import Iterable

# cp1252 covers the German characters we need (umlauts, sharp s, euro). These entries exist for
# the typographic characters a template may use that cp1252 places in the 0x80-0x9F range.
WINANSI_OVERRIDES = {
    "\u201e": 0x84,   # double low quote
    "\u201c": 0x93,   # left double quote
    "\u201d": 0x94,   # right double quote
    "\u201a": 0x82,   # single low quote
    "\u2013": 0x96,   # en dash
    "\u2014": 0x97,   # em dash
    "\u20ac": 0x80,   # euro sign
    "\u2026": 0x85,   # ellipsis
}


def encode_winansi(text: str) -> bytes:
    """Encode to WinAnsi bytes, substituting '?' for anything unmappable."""
    out = bytearray()
    for char in text:
        if char in WINANSI_OVERRIDES:
            out.append(WINANSI_OVERRIDES[char])
            continue
        try:
            out.extend(char.encode("cp1252"))
        except UnicodeEncodeError:
            out.extend(b"?")
    return bytes(out)


def escape_pdf_text(raw: bytes) -> bytes:
    """Escape the three characters that are special inside a PDF literal string."""
    return raw.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")


def build_content_stream(lines: Iterable[str], *, font_size: int = 9, leading: int = 12,
                         start_y: int = 800, margin: int = 40) -> bytes:
    """Eine einspaltige, linksbündige Seite; jede Zeile ein eigener Tj-Operator."""
    parts = ["BT", f"/F1 {font_size} Tf", f"{leading} TL", f"1 0 0 1 {margin} {start_y} Tm"]
    first = True
    for line in lines:
        if not first:
            parts.append("T*")
        first = False
        literal = escape_pdf_text(encode_winansi(line)).decode("latin-1")
        parts.append(f"({literal}) Tj")
    parts.append("ET")
    return "\n".join(parts).encode("latin-1", errors="replace")


def write_pdf(path, lines: Iterable[str], *, compress: bool = True) -> Path:
    """Write a minimal but structurally valid single-page PDF containing a text layer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    content = build_content_stream(lines)
    if compress:
        stream_data = zlib.compress(content, 6)
        filter_entry = b" /Filter /FlateDecode"
    else:
        stream_data = content
        filter_entry = b""

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream_data)).encode() + filter_entry + b" >>\nstream\n"
        + stream_data + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    ]

    buffer = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(buffer))
        buffer += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_offset = len(buffer)
    count = len(objects) + 1
    buffer += f"xref\n0 {count}\n".encode()
    buffer += b"0000000000 65535 f \n"
    for offset in offsets:
        buffer += f"{offset:010d} 00000 n \n".encode()
    buffer += f"trailer\n<< /Size {count} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode()

    path.write_bytes(bytes(buffer))
    return path


def render_scan_style(text_lines: list[str], path, *, width: int = 1240, height: int = 1754,
                      noisy: bool = True, seed: int = 0) -> bool:
    """Rasterise text to a degraded image so the OCR path has something realistic to read.

    Returns False when no imaging stack is available; the caller records the skip instead of
    failing the run.
    """
    try:
        from PIL import Image, ImageDraw, ImageFilter
    except ImportError:
        return False

    import random

    rng = random.Random(seed)
    base = rng.randint(238, 250) if noisy else 250
    image = Image.new("L", (width, height), color=base)
    draw = ImageDraw.Draw(image)
    y = 90
    for line in text_lines:
        draw.text((100, y), line, fill=rng.randint(15, 60) if noisy else 20)
        y += 26
        if y > height - 60:
            break
    if noisy:
        image = image.rotate(rng.uniform(-0.6, 0.6), resample=Image.BILINEAR, fillcolor=base)
        image = image.filter(ImageFilter.GaussianBlur(radius=0.6))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return True
