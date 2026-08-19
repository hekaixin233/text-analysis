"""Command-line interface for corpus analysis."""

import argparse
from pathlib import Path

from .analyzer import SentenceThresholds, analyze_text
from .reader import discover_text_files, read_text
from .reporting import save_reports
from .tokenizer import DEFAULT_STOPWORDS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="分析文本词频及短、中、长句分布")
    parser.add_argument("input", nargs="?", type=Path, default=Path("text"), help="文本文件或目录")
    parser.add_argument("-o", "--output", type=Path, default=Path("output"), help="结果目录")
    parser.add_argument("--pattern", default="*.txt", help="递归匹配规则")
    parser.add_argument("--encoding", help="强制指定输入编码")
    parser.add_argument("--short-max", type=int, default=15, help="短句最大字符数")
    parser.add_argument("--medium-max", type=int, default=30, help="中句最大字符数")
    parser.add_argument("--top", type=int, default=30, help="图表展示的高频词数量")
    parser.add_argument("--stopwords", type=Path, help="附加停用词文件（每行一个）")
    parser.add_argument("--min-word-length", type=int, default=2, help="保留词语的最小字符数")
    parser.add_argument("--keep-proper-nouns", action="store_true", help="保留姓名、地名、机构名等专有名词")
    parser.add_argument("-q", "--quiet", action="store_true", help="不逐个打印语料摘要，适合批量分析")
    return parser


def _load_stopwords(path: Path | None) -> set[str]:
    words = set(DEFAULT_STOPWORDS)
    if path:
        content, _ = read_text(path)
        words.update(line.strip() for line in content.splitlines() if line.strip())
    return words


def _print_summary(corpus_name: str, result, output: Path) -> None:
    rows = [
        ("总词数", result.total_words),
        ("不同词数", result.unique_words),
        ("总句数", result.total_sentences),
    ]
    for label in ("短句", "中句", "长句"):
        rows.append((label, result.sentence_distribution[label]))
    print(f"文本分析摘要：{corpus_name}")
    print("-" * 25)
    for label, value in rows:
        print(f"{label:<8} {value:>12,}")
    print(f"结果目录: {output.resolve()}")


def _corpus_name(path: Path, input_path: Path) -> str:
    if input_path.is_file():
        return path.stem
    relative = path.relative_to(input_path).with_suffix("")
    return "__".join(relative.parts)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.top < 1:
        raise SystemExit("--top 必须大于 0")
    if args.min_word_length < 1:
        raise SystemExit("--min-word-length 必须大于 0")
    thresholds = SentenceThresholds(args.short_max, args.medium_max)
    files = discover_text_files(args.input, args.pattern)
    stopwords = _load_stopwords(args.stopwords)
    for path in files:
        corpus_name = _corpus_name(path, args.input)
        content, encoding = read_text(path, args.encoding)
        result = analyze_text(
            content,
            thresholds,
            stopwords,
            min_word_length=args.min_word_length,
            exclude_proper_nouns=not args.keep_proper_nouns,
        )
        corpus_output = args.output / corpus_name
        save_reports(
            result,
            corpus_output,
            args.top,
            thresholds,
            {"file": str(path), "encoding": encoding},
            corpus_name,
            args.min_word_length,
            not args.keep_proper_nouns,
        )
        if not args.quiet:
            _print_summary(corpus_name, result, corpus_output)


if __name__ == "__main__":
    main()
