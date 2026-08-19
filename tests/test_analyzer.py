import pandas as pd

from text_analysis.analyzer import SentenceThresholds, analyze_text, classify_sentence
from text_analysis.authorship import FEATURES, _auc, _select_representative_corpora
from text_analysis.tokenizer import sentence_length, split_sentences
from text_analysis.importer import normalize_filename
from text_analysis.style_words import count_style_words, style_word_rows


def test_sentence_split_and_length():
    assert split_sentences("你好！这是第二句。\n第三句？") == ["你好", "这是第二句", "第三句"]
    assert sentence_length("“你好，world 2！”") == 8


def test_sentence_classification_boundaries():
    thresholds = SentenceThresholds(5, 10)
    assert classify_sentence(5, thresholds) == "短句"
    assert classify_sentence(6, thresholds) == "中句"
    assert classify_sentence(11, thresholds) == "长句"


def test_analysis_returns_all_categories():
    result = analyze_text("春天来了。花开了！", SentenceThresholds(5, 10), stopwords=set())
    assert result.total_sentences == 2
    assert result.sentence_distribution == {"短句": 2, "中句": 0, "长句": 0}
    assert result.total_words > 0
    assert result.total_characters == 7


def test_common_words_single_characters_and_proper_nouns_are_filtered():
    result = analyze_text("克莱恩来到北京，他看到了神秘事件。", stopwords={"看到"})
    assert "克莱恩" not in result.word_frequencies
    assert "北京" not in result.word_frequencies
    assert all(len(word) >= 2 for word in result.word_frequencies)


def test_corpus_filename_normalization():
    assert normalize_filename("[1991-1]《超新星纪元》(内地实体删节版).txt") == "1991-1__超新星纪元__内地实体删节版.txt"
    assert normalize_filename("[1998-1]《西洋》 .txt") == "1998-1__西洋.txt"


def test_auc_reports_perfect_separation_in_either_direction():
    groups = pd.Series(["人类", "人类", "AI", "AI"])
    auc, direction = _auc(pd.Series([1, 2, 3, 4]), groups)
    assert auc == 1
    assert direction == "AI较高"


def test_representative_selection_returns_each_group():
    rows = []
    for group, offset in (("人类", 0), ("AI", 10)):
        for index in range(4):
            row = {"组别": group, "语料ID": f"{group}-{index}", "语料": f"样本{index}"}
            row.update({feature: offset + index for feature in FEATURES})
            rows.append(row)
    selected = _select_representative_corpora(pd.DataFrame(rows), count=2)
    assert selected.groupby("组别").size().to_dict() == {"AI": 2, "人类": 2}
    assert set(selected["组内代表性排名"]) == {1, 2}


def test_style_word_counts_and_rates():
    text = "然而他没有立刻转身。然而，他只是静静看着。"
    counts = count_style_words(text)
    assert counts["然而"] == 2
    assert counts["立刻"] == 1
    rows = style_word_rows(text)
    assert rows[0]["词语"] == "然而"
    assert rows[0]["次数"] == 2
    assert rows[0]["每万字次数"] > 0
