"""Tokenization and sentence splitting policies."""

import re
from collections.abc import Iterable

import jieba.posseg as pseg

SENTENCE_BOUNDARY = re.compile(r"[。！？!?；;]+[”’\"']?|\n+")
VALID_TOKEN = re.compile(r"^[\u3400-\u9fffA-Za-z0-9]+$")
CONTAINS_CJK = re.compile(r"[\u3400-\u9fff]")
DEFAULT_STOPWORDS = frozenset(
    "的 了 和 是 在 我 有 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 自己 这 那 他 她 它 们 与 及 而 被 把 从 对 为 但 又 还 让 地 得 道 将 能 里 下 后 者 内 向 出 过 并 非凡 不是 就是 可以 什么 怎么 这样 那样 这个 那个 因为 所以 如果 已经 只是 还是 只是 可能 知道 觉得 认为 看到 听到 现在 时候 事情 问题 一些 一种 一下 一点 起来 出来 进去 这里 那里 其中 甚至 然后 不过 而且 或者 以及 对于 关于 由于 通过 进行 需要 应该 能够 无法 十分 非常 似乎 仿佛 当然 突然 依然 仍然 他们 她们 它们 我们 你们 大家 对方 彼此 先生 女士 两个 一位 说道 看见".split()
)
PROPER_NOUN_FLAGS = frozenset({"nr", "nrfg", "nrt", "ns", "nt", "nz"})


def split_sentences(text: str) -> list[str]:
    """Split Chinese/English prose into non-empty sentences."""
    return [sentence.strip() for sentence in SENTENCE_BOUNDARY.split(text) if sentence.strip()]


def tokenize(
    text: str,
    stopwords: Iterable[str] = DEFAULT_STOPWORDS,
    *,
    min_length: int = 2,
    exclude_proper_nouns: bool = True,
) -> list[str]:
    """Segment text, excluding common words, single characters and proper nouns."""
    excluded = set(stopwords)
    tokens: list[str] = []
    for pair in pseg.cut(text):
        token = pair.word.strip().lower()
        if (
            not token
            or len(token) < min_length
            or not VALID_TOKEN.fullmatch(token)
            or not CONTAINS_CJK.search(token)
        ):
            continue
        if token in excluded or (exclude_proper_nouns and pair.flag in PROPER_NOUN_FLAGS):
            continue
        tokens.append(token)
    return tokens


def sentence_length(sentence: str) -> int:
    """Count Chinese/alphanumeric characters, excluding punctuation and whitespace."""
    return sum(len(part) for part in re.findall(r"[\u3400-\u9fffA-Za-z0-9]+", sentence))


def character_count(text: str) -> int:
    """Count corpus characters used as the denominator for normalized frequency."""
    return sentence_length(text)
