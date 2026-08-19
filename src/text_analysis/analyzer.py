"""Core, side-effect-free analysis functions."""

from collections import Counter
from dataclasses import asdict, dataclass

from .tokenizer import DEFAULT_STOPWORDS, character_count, sentence_length, split_sentences, tokenize


@dataclass(frozen=True)
class SentenceThresholds:
    short_max: int = 15
    medium_max: int = 30

    def __post_init__(self) -> None:
        if self.short_max < 1 or self.medium_max <= self.short_max:
            raise ValueError("阈值必须满足 1 <= short_max < medium_max")


@dataclass
class AnalysisResult:
    word_frequencies: Counter[str]
    sentence_distribution: Counter[str]
    sentence_lengths: list[int]
    total_words: int
    unique_words: int
    total_sentences: int
    total_characters: int

    def summary(self) -> dict[str, object]:
        return {
            "total_words": self.total_words,
            "unique_words": self.unique_words,
            "total_sentences": self.total_sentences,
            "total_characters": self.total_characters,
            "average_sentence_length": round(
                sum(self.sentence_lengths) / self.total_sentences, 2
            ) if self.total_sentences else 0,
            "sentence_distribution": dict(self.sentence_distribution),
        }


def classify_sentence(length: int, thresholds: SentenceThresholds) -> str:
    if length <= thresholds.short_max:
        return "短句"
    if length <= thresholds.medium_max:
        return "中句"
    return "长句"


def analyze_text(
    text: str,
    thresholds: SentenceThresholds = SentenceThresholds(),
    stopwords: set[str] | frozenset[str] = DEFAULT_STOPWORDS,
    *,
    min_word_length: int = 2,
    exclude_proper_nouns: bool = True,
) -> AnalysisResult:
    words = tokenize(
        text,
        stopwords,
        min_length=min_word_length,
        exclude_proper_nouns=exclude_proper_nouns,
    )
    lengths = [length for sentence in split_sentences(text) if (length := sentence_length(sentence))]
    distribution = Counter(classify_sentence(length, thresholds) for length in lengths)
    for label in ("短句", "中句", "长句"):
        distribution.setdefault(label, 0)
    frequencies = Counter(words)
    return AnalysisResult(
        word_frequencies=frequencies,
        sentence_distribution=distribution,
        sentence_lengths=lengths,
        total_words=len(words),
        unique_words=len(frequencies),
        total_sentences=len(lengths),
        total_characters=character_count(text),
    )
