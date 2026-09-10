from __future__ import annotations

from pathlib import Path

_CSS = (
    ".ferric-alpha-report{font-family:system-ui,-apple-system,BlinkMacSystemFont,"
    '"Segoe UI",sans-serif;color:#1f2937;background:#fff}.ferric-alpha-report '
    "h1{font-size:22px}.ferric-alpha-grid{display:grid;grid-template-columns:"
    "repeat(12,minmax(0,1fr));gap:24px}.ferric-alpha-panel{overflow:hidden}"
    ".ferric-alpha-span-12{grid-column:span 12}.ferric-alpha-span-6{grid-column:"
    "span 6}.ferric-alpha-panel h2{font-size:16px}.ferric-alpha-table-wrap{"
    "overflow-x:auto}.ferric-alpha-table-wrap table{border-collapse:collapse;"
    "width:100%;font-size:12px}.ferric-alpha-table-wrap th,.ferric-alpha-table-wrap "
    "td{border:1px solid #e5e7eb;padding:4px 6px;white-space:nowrap}"
    "@media(max-width:960px){.ferric-alpha-grid{display:block}.ferric-alpha-panel{"
    "margin-bottom:24px}}"
)
_HTML_PREFIX = (
    '<!doctype html><html><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width, initial-scale=1">'
    f"<title>Ferric Alpha Report</title><style>{_CSS}</style></head><body>"
)
_HTML_SUFFIX = "</body></html>"


class RenderedTearSheet:
    __slots__ = ("_format", "_payload", "_width", "_height")

    def __init__(
        self,
        *,
        format: str,
        payload: str | bytes,
        width: int,
        height: int,
    ) -> None:
        object.__setattr__(self, "_format", format)
        object.__setattr__(self, "_payload", payload)
        object.__setattr__(self, "_width", width)
        object.__setattr__(self, "_height", height)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("RenderedTearSheet is immutable")

    @property
    def format(self) -> str:
        return self._format

    @property
    def content(self) -> str | bytes:
        if self._format == "html":
            return f"{_HTML_PREFIX}{self._payload}{_HTML_SUFFIX}"
        return self._payload

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def _repr_html_(self) -> str | None:
        if self._format == "html":
            return str(self._payload)
        return None

    def _repr_svg_(self) -> str | None:
        if self._format == "svg":
            return str(self._payload)
        return None

    def _repr_png_(self) -> bytes | None:
        if self._format == "png":
            return bytes(self._payload)
        return None

    def save(self, path: str | Path) -> None:
        path = Path(path)
        expected = f".{self._format}"
        if path.suffix != expected:
            raise ValueError(f"invalid extension: expected {expected}")
        content = self.content
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content)

    def __repr__(self) -> str:
        return (
            "RenderedTearSheet("
            f"format={self._format!r}, width={self._width}, height={self._height})"
        )
