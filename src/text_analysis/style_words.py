"""Curated, topic-light words useful for exploratory Chinese style comparison."""

from collections import Counter

from .tokenizer import character_count

# The categories document why a word is included; reports intentionally omit them.
STYLE_WORD_GROUPS: dict[str, tuple[str, ...]] = {
    "衔接与转折": (
        "然而", "但是", "不过", "可是", "却是", "而是", "于是", "因此", "所以", "而且",
        "同时", "随后", "接着", "然后", "最后", "直到", "因为", "如果", "虽然", "即使",
        "既然", "只要", "除非", "为了", "以及", "否则", "反而", "另外", "其中", "至于",
    ),
    "程度与判断": (
        "甚至", "几乎", "仿佛", "似乎", "显然", "当然", "确实", "其实", "或许", "大概",
        "可能", "完全", "极其", "十分", "非常", "有些", "有点", "一点", "只是", "尤其",
    ),
    "时间与推进": (
        "突然", "立刻", "马上", "终于", "已经", "仍然", "依然", "渐渐", "再次", "刚才",
        "片刻", "瞬间", "此时", "这时", "那时", "起初", "原本", "开始", "继续", "准备",
    ),
    "方式与叙述动作": (
        "缓缓", "慢慢", "轻轻", "悄悄", "静静", "默默", "猛地", "狠狠", "不禁", "忍不住",
        "看着", "望着", "盯着", "听着", "想着", "觉得", "知道", "意识到", "点点头", "摇摇头",
        "笑了笑", "深吸一口气", "转身", "抬头", "低头", "皱眉", "沉默", "带着", "朝着", "随着",
    ),
}
STYLE_WORDS = tuple(dict.fromkeys(word for words in STYLE_WORD_GROUPS.values() for word in words))


def count_style_words(text: str) -> Counter[str]:
    """Count exact occurrences of the curated style words in a corpus."""
    return Counter({word: count for word in STYLE_WORDS if (count := text.count(word))})


def style_word_rows(text: str) -> list[dict[str, object]]:
    """Return reference-style rows normalized by effective corpus characters."""
    denominator = character_count(text)
    rows: list[dict[str, object]] = []
    for rank, (word, count) in enumerate(count_style_words(text).most_common(), start=1):
        rate = count / denominator if denominator else 0
        rows.append({
            "排名": rank,
            "词语": word,
            "次数": count,
            "每万字次数": round(rate * 10_000, 2),
            "正文占比（%）": round(rate * 100, 3),
        })
    return rows
