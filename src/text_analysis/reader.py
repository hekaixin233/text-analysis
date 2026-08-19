"""Text discovery and robust decoding utilities."""

from pathlib import Path

SUPPORTED_ENCODINGS = ("utf-8-sig", "gb18030", "big5")


def discover_text_files(path: Path, pattern: str = "*.txt") -> list[Path]:
    """Return sorted text files from a file or directory."""
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"输入路径不存在: {path}")
    files = sorted(item for item in path.rglob(pattern) if item.is_file())
    if not files:
        raise FileNotFoundError(f"在 {path} 中未找到匹配 {pattern!r} 的文件")
    return files


def read_text(path: Path, encoding: str | None = None) -> tuple[str, str]:
    """Decode a text file, returning its content and the encoding used."""
    raw = path.read_bytes()
    candidates = (encoding,) if encoding else SUPPORTED_ENCODINGS
    errors: list[str] = []
    for candidate in candidates:
        try:
            return raw.decode(candidate), candidate
        except UnicodeDecodeError as exc:
            errors.append(f"{candidate}: {exc}")
    raise UnicodeError(f"无法解码 {path}; " + "; ".join(errors))


def load_corpus(files: list[Path], encoding: str | None = None) -> tuple[str, dict[str, str]]:
    """Load and concatenate files while retaining detected encoding metadata."""
    contents: list[str] = []
    encodings: dict[str, str] = {}
    for path in files:
        content, used_encoding = read_text(path, encoding)
        contents.append(content)
        encodings[str(path)] = used_encoding
    return "\n".join(contents), encodings
