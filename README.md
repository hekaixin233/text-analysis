# 中文文本分析

一个模块化、可复用的 Python 项目，用于逐个分析中文语料的标准化词频和句长分布，并探索 AI 与人类写作的风格差异。

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

默认递归分析 `text/` 下的 `*.txt`。程序自动尝试 UTF-8、GB18030 和 Big5 编码。

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

AI 与人类写作的探索性比较采用按有效字数一对一匹配，比较词汇集中度、词汇熵、代表语料高频词、“风格词”、句长均值和句长离散程度。对比样本既可按组内中位风格选择，也可从 ChatGPT、Codex、DeepSeek、Gemini、Grok 各选一个家族中心样本，并用多维风格距离选取分布差异较大的人类作品。“风格词”采用可复用词表统计转折衔接、程度判断、时间推进及叙述动作表达，不显示词性。这些特征只能提供风格线索，不能单独作为 AI 写作判定依据。

核心 API 可以直接复用：

```python
from text_analysis import SentenceThresholds, analyze_text

result = analyze_text("这是一段待分析文本。", SentenceThresholds(15, 30))
print(result.word_frequencies)
print(result.sentence_distribution)
```

运行测试：`uv run pytest`。
