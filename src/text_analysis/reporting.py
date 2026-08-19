"""Persist tabular, structured, and publication-ready visual reports."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .analyzer import AnalysisResult, SentenceThresholds

COLORS = {"primary": "#2878B5", "short": "#45A776", "medium": "#F2A65A", "long": "#D95F59"}


def _configure_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.titleweight": "bold",
        "axes.titlesize": 15,
        "figure.facecolor": "white",
    })


def save_reports(
    result: AnalysisResult,
    output_dir: Path,
    top_n: int,
    thresholds: SentenceThresholds,
    metadata: dict[str, object],
    corpus_name: str,
    min_word_length: int = 2,
    exclude_proper_nouns: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    words = pd.DataFrame(result.word_frequencies.most_common(), columns=["词语", "次数"])
    words.insert(0, "排名", range(1, len(words) + 1))
    rates = (
        words["次数"] / result.total_characters
        if result.total_characters
        else pd.Series(0.0, index=words.index)
    )
    words["每万字次数"] = (rates * 10_000).round(2)
    words["正文占比"] = (rates * 100).map(lambda value: f"{value:.2f}%")
    words.to_csv(output_dir / "word_frequencies.csv", index=False, encoding="utf-8-sig")

    sentences = pd.DataFrame(
        [(name, result.sentence_distribution[name]) for name in ("短句", "中句", "长句")],
        columns=["句子类型", "数量"],
    )
    sentences["占比"] = sentences["数量"] / result.total_sentences if result.total_sentences else 0
    sentences.to_csv(output_dir / "sentence_distribution.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"句长（字）": result.sentence_lengths}).to_csv(
        output_dir / "sentence_lengths.csv", index=False, encoding="utf-8-sig"
    )

    payload = result.summary() | {
        "thresholds": {"short_max": thresholds.short_max, "medium_max": thresholds.medium_max},
        "frequency_unit": "每万有效字符出现次数",
        "filters": {
            "minimum_word_length": min_word_length,
            "proper_nouns_excluded": exclude_proper_nouns,
        },
        "metadata": metadata,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _save_charts(result, output_dir, top_n, thresholds, corpus_name)


def _save_charts(
    result: AnalysisResult,
    output_dir: Path,
    top_n: int,
    thresholds: SentenceThresholds,
    corpus_name: str,
) -> None:
    _configure_style()
    top_words = result.word_frequencies.most_common(top_n)
    if top_words:
        labels = [item[0] for item in reversed(top_words)]
        rates = [item[1] / result.total_characters * 10_000 for item in reversed(top_words)]
        fig, axis = plt.subplots(figsize=(10, max(6, top_n * 0.31)))
        colors = sns.color_palette("crest", n_colors=len(labels))
        bars = axis.barh(labels, rates, color=colors)
        axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
        axis.set(
            title=f"{corpus_name}｜高频词 Top {len(labels)}（已过滤常见词和专有名词）",
            xlabel="每万字出现次数",
            ylabel="词语",
        )
        axis.grid(axis="x", alpha=0.25)
        axis.grid(axis="y", visible=False)
        sns.despine(ax=axis, left=True)
        fig.tight_layout()
        fig.savefig(output_dir / "word_frequency.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    labels = ["短句", "中句", "长句"]
    values = [result.sentence_distribution[label] for label in labels]
    fig, axis = plt.subplots(figsize=(8.5, 5.5))
    bars = axis.bar(labels, values, color=[COLORS["short"], COLORS["medium"], COLORS["long"]], width=0.62)
    percentages = [value / result.total_sentences * 100 if result.total_sentences else 0 for value in values]
    axis.bar_label(bars, labels=[f"{value:,}\n({percent:.1f}%)" for value, percent in zip(values, percentages)], padding=5)
    axis.set(title=f"{corpus_name}｜短、中、长句分布", xlabel="句子类型", ylabel="句子数量")
    axis.grid(axis="y", alpha=0.25)
    axis.grid(axis="x", visible=False)
    sns.despine(ax=axis)
    fig.tight_layout()
    fig.savefig(output_dir / "sentence_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    if result.sentence_lengths:
        lengths = np.asarray(result.sentence_lengths)
        display_max = max(thresholds.medium_max + 5, int(np.percentile(lengths, 99.5)))
        displayed = lengths[lengths <= display_max]
        bins = min(60, max(15, int(np.sqrt(len(displayed)))))
        fig, axis = plt.subplots(figsize=(10, 5.8))
        sns.histplot(displayed, bins=bins, color=COLORS["primary"], edgecolor="white", alpha=0.85, ax=axis)
        axis.axvline(thresholds.short_max, color=COLORS["short"], linestyle="--", linewidth=2, label=f"短句上限：{thresholds.short_max} 字")
        axis.axvline(thresholds.medium_max, color=COLORS["long"], linestyle="--", linewidth=2, label=f"中句上限：{thresholds.medium_max} 字")
        axis.set(
            title=f"{corpus_name}｜句子长度直方图",
            xlabel=f"句长（字；图示至 99.5% 分位：{display_max} 字）",
            ylabel="句子数量",
        )
        axis.legend(frameon=False)
        axis.grid(alpha=0.22)
        sns.despine(ax=axis)
        fig.tight_layout()
        fig.savefig(output_dir / "sentence_length_histogram.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
