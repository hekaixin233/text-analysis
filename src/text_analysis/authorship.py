"""Exploratory AI-vs-human stylometry based on lexical and sentence features."""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .comparison import _corpus_identity, _display_name
from .reader import read_text
from .reporting import _configure_style
from .style_words import STYLE_WORDS, count_style_words, style_word_rows

AI_COLOR = "#7A5AF8"
HUMAN_COLOR = "#E58B3A"
GROUP_ORDER = ["人类", "AI"]
AI_FAMILIES = {
    "ChatGPT": "chatgpt",
    "Codex": "codex",
    "DeepSeek": "deepseek",
    "Gemini": "gemini",
    "Grok": "grok",
}
FEATURES = [
    "平均句长",
    "句长变异系数",
    "短句占比（%）",
    "长句占比（%）",
    "Top10词集中度（%）",
    "词汇归一化熵",
    "一次词占比（%）",
    "风格词密度（每万字）",
    "风格词类型覆盖率（%）",
]


def _load_features(output_root: Path, destination: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    word_records: list[pd.DataFrame] = []
    for summary_path in sorted(output_root.rglob("summary.json")):
        if destination in summary_path.parents or "comparison" in summary_path.parts:
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if payload["total_characters"] <= 0 or payload["total_sentences"] <= 0:
            continue
        group, corpus = _corpus_identity(summary_path, output_root)
        authorship = "AI" if group.casefold() == "ai" else "人类"
        corpus_id = f"{group}/{corpus}"
        lengths = pd.read_csv(summary_path.parent / "sentence_lengths.csv", encoding="utf-8-sig").iloc[:, 0].to_numpy(dtype=float)
        words = pd.read_csv(summary_path.parent / "word_frequencies.csv", encoding="utf-8-sig")
        counts = words["次数"].to_numpy(dtype=float)
        probabilities = counts / counts.sum() if counts.sum() else np.array([])
        entropy = float(-(probabilities * np.log(probabilities)).sum() / np.log(len(probabilities))) if len(probabilities) > 1 else 0
        source_text, _ = read_text(Path(payload["metadata"]["file"]), payload["metadata"].get("encoding"))
        style_counts = count_style_words(source_text)
        style_total = sum(style_counts.values())
        distribution = payload["sentence_distribution"]
        total_sentences = payload["total_sentences"]
        mean_length = float(lengths.mean())
        rows.append({
            "语料ID": corpus_id,
            "结果目录": summary_path.parent.relative_to(output_root).as_posix(),
            "语料": _display_name(group, corpus),
            "组别": authorship,
            "有效字数": payload["total_characters"],
            "平均句长": mean_length,
            "句长标准差": float(lengths.std(ddof=0)),
            "句长变异系数": float(lengths.std(ddof=0) / mean_length) if mean_length else 0,
            "句长中位数": float(np.median(lengths)),
            "句长90分位": float(np.percentile(lengths, 90)),
            "短句占比（%）": distribution["短句"] / total_sentences * 100,
            "长句占比（%）": distribution["长句"] / total_sentences * 100,
            "Top10词集中度（%）": counts[:10].sum() / counts.sum() * 100 if counts.sum() else 0,
            "词汇归一化熵": entropy,
            "一次词占比（%）": (counts == 1).sum() / len(counts) * 100 if len(counts) else 0,
            "风格词密度（每万字）": style_total / payload["total_characters"] * 10_000,
            "风格词类型覆盖率（%）": len(style_counts) / len(STYLE_WORDS) * 100,
        })
        rate_column = "每万字次数" if "每万字次数" in words.columns else "每万字出现次数"
        word_records.append(pd.DataFrame({
            "语料ID": corpus_id,
            "组别": authorship,
            "词语": words["词语"],
            "每万字次数": words[rate_column],
        }))
    return pd.DataFrame(rows), pd.concat(word_records, ignore_index=True)


def _length_matched_sample(features: pd.DataFrame) -> pd.DataFrame:
    ai = features[features["组别"] == "AI"].sort_values("有效字数")
    available = features[features["组别"] == "人类"].copy()
    selected: list[pd.Series] = []
    for _, ai_row in ai.iterrows():
        distances = (np.log(available["有效字数"]) - math.log(ai_row["有效字数"])).abs()
        index = distances.idxmin()
        selected.append(available.loc[index])
        available = available.drop(index)
    human = pd.DataFrame(selected)
    return pd.concat([human, ai], ignore_index=True)


def _auc(values: pd.Series, groups: pd.Series) -> tuple[float, str]:
    ai = values[groups == "AI"].to_numpy()
    human = values[groups == "人类"].to_numpy()
    raw = np.mean([(a > h) + 0.5 * (a == h) for a in ai for h in human])
    return (float(raw), "AI较高") if raw >= 0.5 else (float(1 - raw), "人类较高")


def _separability_table(matched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in FEATURES:
        auc, direction = _auc(matched[feature], matched["组别"])
        rows.append({
            "指标": feature,
            "AI中位数": round(matched.loc[matched["组别"] == "AI", feature].median(), 4),
            "人类中位数": round(matched.loc[matched["组别"] == "人类", feature].median(), 4),
            "样本内AUC": round(auc, 3),
            "较高组": direction,
        })
    return pd.DataFrame(rows).sort_values("样本内AUC", ascending=False)


def _plot_feature_distributions(matched: pd.DataFrame, destination: Path) -> None:
    _configure_style()
    metrics = [
        "平均句长",
        "句长变异系数",
        "Top10词集中度（%）",
        "词汇归一化熵",
        "风格词密度（每万字）",
        "风格词类型覆盖率（%）",
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 13))
    palette = {"人类": HUMAN_COLOR, "AI": AI_COLOR}
    for axis, metric in zip(axes.flat, metrics):
        sns.boxplot(data=matched, x="组别", y=metric, order=GROUP_ORDER, hue="组别", palette=palette, width=0.5, showfliers=False, legend=False, ax=axis)
        sns.stripplot(data=matched, x="组别", y=metric, order=GROUP_ORDER, color="#333333", alpha=0.65, jitter=0.16, size=4, ax=axis)
        axis.set(title=metric, xlabel="", ylabel="")
        axis.grid(axis="y", alpha=0.22)
        axis.grid(axis="x", visible=False)
        sns.despine(ax=axis)
    fig.suptitle(f"AI 与人类写作特征分布（按字数 1:1 匹配，各 {sum(matched['组别'] == 'AI')} 篇）", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(destination / "ai_human_feature_distributions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_separability(table: pd.DataFrame, destination: Path) -> None:
    _configure_style()
    plot = table.sort_values("样本内AUC")
    colors = [AI_COLOR if direction == "AI较高" else HUMAN_COLOR for direction in plot["较高组"]]
    fig, axis = plt.subplots(figsize=(10, 6))
    bars = axis.barh(
        plot["指标"],
        plot["样本内AUC"] - 0.5,
        left=0.5,
        color=colors,
    )
    for bar, auc in zip(bars, plot["样本内AUC"]):
        axis.text(auc + 0.006, bar.get_y() + bar.get_height() / 2, f"{auc:.2f}", va="center")
    axis.axvline(0.5, color="#555555", linestyle="--", linewidth=1.5, label="随机水平 0.50")
    axis.set(xlim=(0.49, 1.02), title="单项指标的样本内区分度（探索性 AUC）", xlabel="AUC（越接近 1，当前样本分离越明显）", ylabel="")
    axis.legend(frameon=False, loc="lower right")
    axis.grid(axis="x", alpha=0.22)
    axis.grid(axis="y", visible=False)
    sns.despine(ax=axis, left=True)
    fig.tight_layout()
    fig.savefig(destination / "single_feature_separability.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _select_representative_corpora(matched: pd.DataFrame, count: int = 3) -> pd.DataFrame:
    """Select corpora nearest each group's multivariate median style profile."""
    selected: list[pd.DataFrame] = []
    for group in GROUP_ORDER:
        subset = matched[matched["组别"] == group].copy()
        values = subset[FEATURES].astype(float)
        median = values.median()
        scale = (values.quantile(0.75) - values.quantile(0.25)).replace(0, 1)
        subset["代表性距离"] = (((values - median) / scale) ** 2).mean(axis=1) ** 0.5
        subset["组内代表性排名"] = subset["代表性距离"].rank(method="first").astype(int)
        selected.append(subset.nsmallest(min(count, len(subset)), "代表性距离"))
    return pd.concat(selected, ignore_index=True)


def _robust_style_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame[FEATURES].astype(float)
    scale = (values.quantile(0.75) - values.quantile(0.25)).replace(0, 1)
    return (values - values.median()) / scale


def _select_cross_model_and_distinctive_corpora(
    features: pd.DataFrame,
    count: int = 5,
) -> pd.DataFrame:
    """Pick one medoid-like sample per AI family and diverse human samples."""
    ai = features[features["组别"] == "AI"].copy()
    ai_selected: list[pd.Series] = []
    for family, prefix in AI_FAMILIES.items():
        family_rows = ai[ai["语料ID"].str.split("/").str[-1].str.casefold().str.startswith(prefix)]
        if family_rows.empty:
            continue
        coordinates = _robust_style_coordinates(family_rows)
        distance = (coordinates**2).mean(axis=1) ** 0.5
        row = family_rows.loc[distance.idxmin()].copy()
        row["选择依据"] = f"{family} 家族中心样本"
        row["显示标签"] = f"{family} · {str(row['语料ID']).split('/')[-1]}"
        ai_selected.append(row)
    ai_frame = pd.DataFrame(ai_selected).head(count)
    ai_frame["组内代表性排名"] = range(1, len(ai_frame) + 1)

    human = features[(features["组别"] == "人类") & (features["有效字数"] >= 3_000)].copy()
    coordinates = _robust_style_coordinates(human)
    remaining = list(human.index)
    chosen: list[int] = []
    while remaining and len(chosen) < count:
        if not chosen:
            distances = (coordinates.loc[remaining] ** 2).mean(axis=1) ** 0.5
        else:
            distances = pd.Series(
                {
                    index: min(
                        float(np.sqrt(((coordinates.loc[index] - coordinates.loc[other]) ** 2).mean()))
                        for other in chosen
                    )
                    for index in remaining
                }
            )
        selected_index = int(distances.idxmax())
        chosen.append(selected_index)
        remaining.remove(selected_index)
    human_frame = human.loc[chosen].copy()
    human_frame["选择依据"] = ["偏离组内中心最明显"] + ["与已选作品风格距离最大"] * (len(human_frame) - 1)
    human_frame["显示标签"] = human_frame["语料"].str.split("/", n=1).str[-1]
    human_frame["组内代表性排名"] = range(1, len(human_frame) + 1)
    return pd.concat([human_frame, ai_frame], ignore_index=True)


def _plot_representative_word_frequencies(
    words: pd.DataFrame,
    representatives: pd.DataFrame,
    destination: Path,
    top_n: int = 10,
    filename: str = "representative_corpus_word_frequencies.png",
    title: str = "AI 与人类代表语料高频词对比",
) -> None:
    """Plot directly comparable top-word bars for representative corpora."""
    _configure_style()
    rows = len(representatives[representatives["组别"] == "人类"])
    columns = 2
    fig, axes = plt.subplots(rows, columns, figsize=(14, max(8, rows * 4.2)), squeeze=False)
    global_max = 0.0
    panels: list[tuple[object, pd.DataFrame, pd.Series]] = []
    for column, group in enumerate(GROUP_ORDER):
        group_representatives = representatives[representatives["组别"] == group].sort_values("组内代表性排名")
        for row, (_, corpus) in enumerate(group_representatives.iterrows()):
            top = (
                words[words["语料ID"] == corpus["语料ID"]]
                .nlargest(top_n, "每万字次数")
                .sort_values("每万字次数")
            )
            global_max = max(global_max, float(top["每万字次数"].max()))
            panels.append((axes[row, column], top, corpus))

    for axis, top, corpus in panels:
        color = AI_COLOR if corpus["组别"] == "AI" else HUMAN_COLOR
        bars = axis.barh(top["词语"], top["每万字次数"], color=color, alpha=0.9)
        axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
        short_name = corpus.get("显示标签", str(corpus["语料"]).split("/", 1)[-1])
        axis.set(
            title=f"{corpus['组别']}｜{short_name}",
            xlabel="每万字次数",
            ylabel="",
            xlim=(0, global_max * 1.15),
        )
        axis.grid(axis="x", alpha=0.22)
        axis.grid(axis="y", visible=False)
        sns.despine(ax=axis, left=True)
    fig.suptitle(
        f"{title}（各 {rows} 篇，每篇 Top {top_n}）",
        fontsize=16,
        fontweight="bold",
        y=0.998,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(destination / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _representative_style_word_analysis(
    output_root: Path,
    representatives: pd.DataFrame,
    destination: Path,
    top_n: int = 15,
    table_filename: str = "representative_style_word_frequencies.csv",
    chart_filename: str = "representative_style_word_frequencies.png",
    title: str = "AI 与人类代表语料“风格词”对比",
) -> pd.DataFrame:
    """Create reference-style tables and comparable plots for style words."""
    frames: list[pd.DataFrame] = []
    for _, corpus in representatives.iterrows():
        result_directory = str(corpus.get("结果目录", corpus["语料ID"]))
        summary_path = output_root.joinpath(*result_directory.split("/"), "summary.json")
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        source_path = Path(payload["metadata"]["file"])
        text, _ = read_text(source_path, payload["metadata"].get("encoding"))
        frame = pd.DataFrame(style_word_rows(text))
        if frame.empty:
            continue
        frame.insert(0, "语料", corpus["语料"])
        frame.insert(0, "组别", corpus["组别"])
        frame.insert(0, "语料ID", corpus["语料ID"])
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(destination / table_filename, index=False, encoding="utf-8-sig")

    rows = representatives.groupby("组别").size().min()
    fig, axes = plt.subplots(rows, 2, figsize=(14, max(10, rows * 5.2)), squeeze=False)
    panels: list[tuple[object, pd.DataFrame, pd.Series]] = []
    global_max = 0.0
    for column, group in enumerate(GROUP_ORDER):
        group_rows = representatives[representatives["组别"] == group].sort_values("组内代表性排名").head(rows)
        for row, (_, corpus) in enumerate(group_rows.iterrows()):
            top = (
                combined[combined["语料ID"] == corpus["语料ID"]]
                .nsmallest(top_n, "排名")
                .sort_values("每万字次数")
            )
            global_max = max(global_max, float(top["每万字次数"].max()))
            panels.append((axes[row, column], top, corpus))
    for axis, top, corpus in panels:
        color = AI_COLOR if corpus["组别"] == "AI" else HUMAN_COLOR
        bars = axis.barh(top["词语"], top["每万字次数"], color=color, alpha=0.9)
        axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
        short_name = corpus.get("显示标签", str(corpus["语料"]).split("/", 1)[-1])
        axis.set(
            title=f"{corpus['组别']}｜{short_name}",
            xlabel="每万字次数",
            ylabel="",
            xlim=(0, global_max * 1.16),
        )
        axis.grid(axis="x", alpha=0.22)
        axis.grid(axis="y", visible=False)
        sns.despine(ax=axis, left=True)
    fig.suptitle(
        f"{title}（各 {rows} 篇，每篇 Top {top_n}）",
        fontsize=16,
        fontweight="bold",
        y=0.998,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(destination / chart_filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return combined


def _word_difference(words: pd.DataFrame, matched: pd.DataFrame, destination: Path) -> pd.DataFrame:
    selected_ids = set(matched["语料ID"])
    words = words[words["语料ID"].isin(selected_ids)]
    group_sizes = matched.groupby("组别")["语料ID"].nunique()
    presence = words.groupby(["组别", "词语"])["语料ID"].nunique().unstack(0, fill_value=0)
    minimum_ai = max(3, math.ceil(group_sizes["AI"] * 0.2))
    minimum_human = max(3, math.ceil(group_sizes["人类"] * 0.2))
    eligible = presence[(presence.get("AI", 0) >= minimum_ai) & (presence.get("人类", 0) >= minimum_human)].index
    matrix = words[words["词语"].isin(eligible)].pivot_table(index="词语", columns="语料ID", values="每万字次数", fill_value=0)
    ai_columns = matched.loc[matched["组别"] == "AI", "语料ID"]
    human_columns = matched.loc[matched["组别"] == "人类", "语料ID"]
    comparison = pd.DataFrame({
        "AI平均每万字次数": matrix.reindex(columns=ai_columns, fill_value=0).mean(axis=1),
        "人类平均每万字次数": matrix.reindex(columns=human_columns, fill_value=0).mean(axis=1),
    })
    comparison["AI对人类log2比值"] = np.log2((comparison["AI平均每万字次数"] + 0.5) / (comparison["人类平均每万字次数"] + 0.5))
    comparison = comparison.reset_index().sort_values("AI对人类log2比值", ascending=False)
    comparison.to_csv(destination / "ai_human_word_difference.csv", index=False, encoding="utf-8-sig")

    extremes = pd.concat([comparison.head(10), comparison.tail(10)]).drop_duplicates("词语").sort_values("AI对人类log2比值")
    _configure_style()
    fig, axis = plt.subplots(figsize=(10, 7))
    colors = [AI_COLOR if value > 0 else HUMAN_COLOR for value in extremes["AI对人类log2比值"]]
    axis.barh(extremes["词语"], extremes["AI对人类log2比值"], color=colors)
    axis.axvline(0, color="#555555", linewidth=1)
    axis.set(title="AI 与人类语料的高频词倾向差异", xlabel="平均每万字次数的 log2 比值（左：人类较高；右：AI较高）", ylabel="")
    axis.grid(axis="x", alpha=0.22)
    axis.grid(axis="y", visible=False)
    sns.despine(ax=axis, left=True)
    fig.tight_layout()
    fig.savefig(destination / "ai_human_word_difference.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return comparison


def _write_interpretation(destination: Path, features: pd.DataFrame, matched: pd.DataFrame, separability: pd.DataFrame) -> None:
    best = separability.iloc[0]
    destination.joinpath("interpretation.md").write_text(
        "# 词频和句长能否鉴别 AI 写作？\n\n"
        f"本次包含 {sum(features['组别'] == 'AI')} 篇已知 AI 语料和 {sum(features['组别'] == '人类')} 篇人类语料；"
        f"探索图使用按有效字数一对一匹配的 {sum(matched['组别'] == 'AI')}+{sum(matched['组别'] == '人类')} 篇样本。\n\n"
        f"当前样本中区分度最高的单项指标是“{best['指标']}”（AUC={best['样本内AUC']:.3f}，{best['较高组']}）。"
        "这表示该指标在这批材料中存在差异，不等于它能识别未知文本。\n\n"
        "## 结论\n\n"
        "本样本的句长指标提供了很强的组间线索，但仅靠词频和句子长度仍不能可靠鉴别未知 AI 写作。"
        "词频强烈受题材、提示词和人物设定影响；句长受作者、体裁、章节格式和后期编辑影响。"
        "本数据中的 AI 文本来自有限模型并具有相近网文/科幻任务，人类组主要为特定科幻作者，存在明显来源与题材混杂。"
        "任何样本内 AUC 都可能高估面对新模型、新作者和人工润色文本时的效果。\n\n"
        "若要构建检测器，应扩大不同模型、作者和题材的数据，按作者与提示词划分训练/测试集，并结合句法、重复片段、语义一致性等特征进行外部验证。\n",
        encoding="utf-8",
    )


def build_authorship_report(output_root: Path, destination: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    destination.mkdir(parents=True, exist_ok=True)
    features, words = _load_features(output_root, destination)
    matched = _length_matched_sample(features)
    separability = _separability_table(matched)
    representatives = _select_representative_corpora(matched)
    diverse_corpora = _select_cross_model_and_distinctive_corpora(features)
    features.to_csv(destination / "all_corpus_features.csv", index=False, encoding="utf-8-sig")
    matched.to_csv(destination / "length_matched_features.csv", index=False, encoding="utf-8-sig")
    separability.to_csv(destination / "single_feature_separability.csv", index=False, encoding="utf-8-sig")
    representatives.to_csv(destination / "representative_corpora.csv", index=False, encoding="utf-8-sig")
    diverse_corpora.to_csv(destination / "cross_model_distinctive_corpora.csv", index=False, encoding="utf-8-sig")
    _plot_feature_distributions(matched, destination)
    _plot_separability(separability, destination)
    _plot_representative_word_frequencies(words, representatives, destination)
    _representative_style_word_analysis(output_root, representatives, destination)
    _plot_representative_word_frequencies(
        words,
        diverse_corpora,
        destination,
        filename="cross_model_distinctive_word_frequencies.png",
        title="不同 AI 模型与特色人类语料高频词对比",
    )
    _representative_style_word_analysis(
        output_root,
        diverse_corpora,
        destination,
        table_filename="cross_model_distinctive_style_word_frequencies.csv",
        chart_filename="cross_model_distinctive_style_word_frequencies.png",
        title="不同 AI 模型与特色人类语料“风格词”对比",
    )
    _word_difference(words, matched, destination)
    _write_interpretation(destination, features, matched, separability)
    return features, separability


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="探索词频与句长对 AI/人类写作的区分能力")
    parser.add_argument("input", nargs="?", type=Path, default=Path("output"))
    parser.add_argument("-o", "--output", type=Path, default=Path("output/comparison/ai_vs_human"))
    args = parser.parse_args(argv)
    features, separability = build_authorship_report(args.input, args.output)
    print(f"已分析 {len(features)} 个非空语料；最高样本内 AUC={separability['样本内AUC'].max():.3f}")


if __name__ == "__main__":
    main()
