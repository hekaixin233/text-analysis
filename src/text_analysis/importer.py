"""Reproducible importer for the Liu Cixin corpus hosted on GitHub."""

import argparse
import json
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

REPOSITORY = "VeejaLiu/ScienceFictionCollection"
SOURCE_DIRECTORY = "001 - 刘慈欣(Cixin Liu)"
BRANCH = "master"
API_URL = f"https://api.github.com/repos/{REPOSITORY}/contents/{quote(SOURCE_DIRECTORY)}?ref={BRANCH}"
SOURCE_URL = f"https://github.com/{REPOSITORY}/tree/{BRANCH}/{quote(SOURCE_DIRECTORY)}"
NAME_PATTERN = re.compile(r"^\[(?P<date>[^]]+)]《(?P<title>[^》]+)》(?P<variant>.*)\.txt$", re.IGNORECASE)
INVALID_FILENAME = re.compile(r"[<>:\"/\\|?*\x00-\x1f]")


def normalize_filename(original: str) -> str:
    """Convert the repository's display-oriented names to stable file names."""
    name = unicodedata.normalize("NFKC", original).strip()
    match = NAME_PATTERN.match(name)
    if not match:
        stem = _clean_component(Path(name).stem)
        return f"{stem}.txt"
    parts = [
        _clean_component(match.group("date")),
        _clean_component(match.group("title")),
    ]
    variant = match.group("variant").strip().strip("()（） ")
    if variant:
        parts.append(_clean_component(variant))
    return "__".join(filter(None, parts)) + ".txt"


def _clean_component(value: str) -> str:
    value = INVALID_FILENAME.sub("-", value)
    value = re.sub(r"[《》\[\]()（）]", "", value)
    value = re.sub(r"[\s　]+", "-", value.strip())
    value = re.sub(r"[-_]{2,}", "-", value)
    return value.strip(" .-_")


def _request_json(url: str) -> object:
    request = Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "text-analysis-corpus-importer"})
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def _download(url: str) -> bytes:
    encoded_url = quote(url, safe=":/?=&%")
    request = Request(encoded_url, headers={"User-Agent": "text-analysis-corpus-importer"})
    with urlopen(request, timeout=120) as response:
        return response.read()


def import_corpus(target: Path, workers: int = 8) -> list[dict[str, str]]:
    """Download all TXT files and return a source-to-local manifest."""
    entries = _request_json(API_URL)
    if not isinstance(entries, list):
        raise RuntimeError("GitHub API 未返回目录文件列表")
    source_files = [entry for entry in entries if entry.get("type") == "file" and entry["name"].lower().endswith(".txt")]
    normalized = [normalize_filename(entry["name"]) for entry in source_files]
    if len(normalized) != len(set(normalized)):
        raise ValueError("规范化后出现重名文件，请检查源目录")

    target.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        contents = list(pool.map(lambda entry: _download(entry["download_url"]), source_files))

    manifest: list[dict[str, str]] = []
    for entry, local_name, content in zip(source_files, normalized, contents):
        (target / local_name).write_bytes(content)
        manifest.append({
            "original_name": entry["name"],
            "normalized_name": local_name,
            "sha": entry["sha"],
            "source_url": entry["html_url"],
        })
    manifest.sort(key=lambda item: item["normalized_name"])
    (target / "source_manifest.json").write_text(
        json.dumps({"repository": REPOSITORY, "directory": SOURCE_DIRECTORY, "branch": BRANCH, "source_url": SOURCE_URL, "files": manifest}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="下载并规范化刘慈欣科幻作品语料")
    parser.add_argument("-o", "--output", type=Path, default=Path("text/liu_cixin"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers 必须大于 0")
    manifest = import_corpus(args.output, args.workers)
    print(f"已导入 {len(manifest)} 个文本文件至 {args.output.resolve()}")


if __name__ == "__main__":
    main()
