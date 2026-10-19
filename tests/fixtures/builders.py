"""Builders for small binary test files (no external fixtures, fully deterministic)."""

from __future__ import annotations

import io

import docx


def _esc(line: str) -> str:
    return line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(pages: list[str]) -> bytes:
    """Build a minimal valid PDF; each page's text may contain newlines. An empty string = blank page."""
    n = len(pages)
    objs: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        ("<< /Type /Pages /Kids [" + " ".join(f"{4 + 2 * i} 0 R" for i in range(n)) + f"] /Count {n} >>").encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, text in enumerate(pages):
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {5 + 2 * i} 0 R "
            "/Resources << /Font << /F1 3 0 R >> >> >>".encode()
        )
        if text:
            ops = ["BT /F1 12 Tf 14 TL 72 720 Td"] + [f"({_esc(ln)}) Tj T*" for ln in text.split("\n")] + ["ET"]
            stream = "\n".join(ops).encode("latin-1")
        else:
            stream = b""
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def make_docx(
    heading: str = "Travel Policy",
    paragraphs: tuple[str, ...] = ("Employees must book economy class.",),
    table_rows: tuple[tuple[str, ...], ...] = (("Item", "Limit"), ("Meals", "60 USD")),
) -> bytes:
    """Build a DOCX with one Heading 1, body paragraphs, and a table."""
    document = docx.Document()
    document.add_heading(heading, level=1)
    for text in paragraphs:
        document.add_paragraph(text)
    if table_rows:
        table = document.add_table(rows=len(table_rows), cols=len(table_rows[0]))
        for r, row in enumerate(table_rows):
            for c, value in enumerate(row):
                table.cell(r, c).text = value
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()
