"""Build reference-style tables and cross-corpus comparison charts."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .reporting import COLORS, _configure_style

WORD_RATE_COLUMNS = ("每万字次数", "每万字出现次数")


def _corpus_identity(summary_path: Path, output_root: Path) -> tuple[str, str]:
    relative = summary_path.parent.relative_to(output_root)
    parts = relative.parts
    if len(parts) == 1:
        return "原始语料", parts[0]
    return parts[0], "/".join(parts[1:])


def _display_name(group: str, corpus: str) -> str:
    readable = corpus.replace("__", " · ").replace("_", " ")
    return readable if group == "原始语料" else f"{group}/{readable}"


def _reference_word_table(path: Path, total_characters: int) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    count_column = "次数" if "次数" in frame.columns else "频次"
    rates = (
        frame[count_column] / total_characters * 10_000
        if total_characters
        else pd.Series(0.0, index=frame.index)
    )
    reference = pd.DataFrame({
        "排名": range(1, len(frame) + 1),
        "词语": frame["词语"],
        "次数": frame[count_column],
        "每万字次数": rates.round(2),
        "正文占比": (rates / 100).map(lambda value: f"{value:.2f}%"),
    })
    reference.to_csv(path, index=False, encoding="utf-8-sig")
    return reference


def build_comparison(output_root: Path, destination: Path, top_n: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect all completed analyses into comparable tables and charts."""
    summary_rows: list[dict[str, object]] = []
    word_rows: list[pd.DataFrame] = []
    summaries = sorted(
        path for path in output_root.rglob("summary.json")
        if destination not in path.parents and path.parent != destination
    )
    if not summaries:
        raise FileNotFoundError(f"在 {output_root} 中未找到分析结果")

    for summary_path in summaries:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        group, corpus = _corpus_identity(summary_path, output_root)
        distribution = payload["sentence_distribution"]
        sentences = payload["total_sentences"]
        summary_rows.append({
            "语料组": group,
            "语料": corpus,
            "有效字数": payload["total_characters"],
            "过滤后词数": payload["total_words"],
            "不同词数": payload["unique_words"],
            "句子数": sentences,
            "平均句长": payload["average_sentence_length"],
            "短句占比（%）": round(distribution["短句"] / sentences * 100, 2) if sentences else 0,
            "中句占比（%）": round(distribution["中句"] / sentences * 100, 2) if sentences else 0,
            "长句占比（%）": round(distribution["长句"] / sentences * 100, 2) if sentences else 0,
        })
        word_path = summary_path.parent / "word_frequencies.csv"
        if word_path.exists():
            words = _reference_word_table(word_path, payload["total_characters"])
            top = words.head(top_n).copy()
            top.insert(0, "语料", corpus)
            top.insert(0, "语料组", group)
            top["正文占比（%）"] = (top["每万字次数"] / 100).round(2)
            top = top.drop(columns=["正文占比"])
            word_rows.append(top)

    summary_frame = pd.DataFrame(summary_rows).sort_values(["语料组", "语料"])
    word_frame = pd.concat(word_rows, ignore_index=True)
    destination.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(destination / "corpus_summary.csv", index=False, encoding="utf-8-sig")
    word_frame.to_csv(destination / "top_words_comparison.csv", index=False, encoding="utf-8-sig")
    _save_sentence_profile(summary_frame, destination)
    _save_word_heatmap(word_frame, destination)
    return summary_frame, word_frame


def _save_sentence_profile(frame: pd.DataFrame, destination: Path) -> None:
    _configure_style()
    plot = frame[frame["句子数"] > 0].copy()
    plot["显示名称"] = [_display_name(group, corpus) for group, corpus in zip(plot["语料组"], plot["语料"])]
    plot = plot.sort_values(["长句占比（%）", "显示名称"])
    height = max(9, len(plot) * 0.29)
    fig, axis = plt.subplots(figsize=(13, height))
    left = np.zeros(len(plot))
    for column, label, color in (
        ("短句占比（%）", "短句 ≤15字", COLORS["short"]),
        ("中句占比（%）", "中句 16–30字", COLORS["medium"]),
        ("长句占比（%）", "长句 >30字", COLORS["long"]),
    ):
        values = plot[column].to_numpy()
        axis.barh(plot["显示名称"], values, left=left, label=label, color=color, height=0.72)
        left += values
    axis.set(title="各语料短、中、长句构成对比", xlabel="句子占比（%）", ylabel="")
    axis.set_xlim(0, 100)
    axis.legend(ncols=3, loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False)
    axis.grid(axis="x", alpha=0.22)
    axis.grid(axis="y", visible=False)
    sns.despine(ax=axis, left=True)
    fig.tight_layout()
    fig.savefig(destination / "sentence_profile_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_word_heatmap(frame: pd.DataFrame, destination: Path, word_count: int = 20) -> None:
    _configure_style()
    ranked = (
        frame.groupby("词语")
        .agg(入选语料数=("语料", "nunique"), 平均每万字次数=("每万字次数", "mean"))
        .sort_values(["入选语料数", "平均每万字次数"], ascending=False)
        .head(word_count)
    )
    selected = frame[frame["词语"].isin(ranked.index)].copy()
    selected["显示名称"] = [_display_name(group, corpus) for group, corpus in zip(selected["语料组"], selected["语料"])]
    matrix = selected.pivot_table(index="显示名称", columns="词语", values="每万字次数", fill_value=0)
    matrix = matrix.reindex(columns=ranked.index)
    matrix = matrix.loc[matrix.mean(axis=1).sort_values().index]
    height = max(10, len(matrix) * 0.3)
    fig, axis = plt.subplots(figsize=(16, height))
    sns.heatmap(
        matrix,
        cmap="YlGnBu",
        robust=True,
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": "每万字次数"},
        ax=axis,
    )
    axis.set(title=f"各语料高频词对比（跨语料最常入选的 {len(matrix.columns)} 个词）", xlabel="词语", ylabel="")
    axis.tick_params(axis="x", rotation=35)
    axis.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(destination / "top_words_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="生成跨语料可比表格和可视化")
    parser.add_argument("input", nargs="?", type=Path, default=Path("output"), help="分析结果根目录")
    parser.add_argument("-o", "--output", type=Path, default=Path("output/comparison"))
    parser.add_argument("--top", type=int, default=30, help="每个语料纳入比较的高频词数")
    args = parser.parse_args(argv)
    if args.top < 1:
        raise SystemExit("--top 必须大于 0")
    summaries, words = build_comparison(args.input, args.output, args.top)
    print(f"已比较 {len(summaries)} 个语料、{len(words)} 条高频词记录")


if __name__ == "__main__":
    main()
