# 中文文本分析

一个模块化、可复用的 Python 项目，用于逐个分析中文语料的标准化词频和句长分布，并生成 CSV、JSON 与 PNG 报告。

## 快速开始

```powershell
uv sync
uv run text-analysis
```

从项目所用的公开 GitHub 目录重新导入并规范化刘慈欣语料：

```powershell
uv run import-liu-cixin
uv run text-analysis text/liu_cixin -o output/liu_cixin --quiet
uv run text-compare output -o output/comparison
uv run text-authorship output -o output/comparison/ai_vs_human
```

默认递归分析 `text/` 下的 `*.txt`，每个语料的结果分别写入 `output/<语料名>/`。程序自动尝试 UTF-8、GB18030 和 Big5 编码。

自定义输入、阈值和高频词数量：

```powershell
uv run text-analysis text -o output --short-max 15 --medium-max 30 --top 30
```

附加自己的停用词（每行一个词）：

```powershell
uv run text-analysis text --stopwords stopwords.txt
```

## 统计口径

- 词频：使用 jieba 词性分词，默认排除单字、常见停用词以及姓名、地名、机构名等专有名词。
- 标准化频率：`词语频次 / 语料有效字符数 × 10000`，即每万字出现次数。
- 句子：以中文或英文句号、问号、感叹号、分号及换行为边界。
- 句长：统计句内中文、字母和数字的字符数，不计空白及标点。
- 默认分类：短句不超过 15 字，中句 16–30 字，长句超过 30 字。

## 输出文件

- `word_frequencies.csv`：参考式词频表（排名、词语、次数、每万字次数、正文占比）
- `sentence_distribution.csv`：三类句子的数量和占比
- `sentence_lengths.csv`：每个句子的字符长度
- `summary.json`：摘要、阈值、源文件及编码信息
- `word_frequency.png`：高频词图
- `sentence_distribution.png`：句型分布图
- `sentence_length_histogram.png`：句长直方图（展示至 99.5% 分位，避免极端值压缩主体）

跨语料比较结果位于 `output/comparison/`：

- `corpus_summary.csv`：字数、词数、句数、平均句长及三类句子占比
- `top_words_comparison.csv`：所有语料 Top 词的统一长表
- `top_words_heatmap.png`：标准化高频词热力图
- `sentence_profile_comparison.png`：短、中、长句 100% 堆叠对比图

AI 与人类写作的探索性比较位于 `output/comparison/ai_vs_human/`。分析采用按有效字数一对一匹配，比较词汇集中度、词汇熵、句长均值和句长离散程度；这些特征只能提供风格线索，不能单独作为 AI 写作判定依据。

核心 API 可以直接复用：

```python
from text_analysis import SentenceThresholds, analyze_text

result = analyze_text("这是一段待分析文本。", SentenceThresholds(15, 30))
print(result.word_frequencies)
print(result.sentence_distribution)
```

运行测试：`uv run pytest`。
