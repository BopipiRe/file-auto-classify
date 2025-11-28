import math
import re
from collections import Counter
from typing import Iterable, List, Set

try:
    import jieba

    _HAS_JIEBA = True
except Exception:
    _HAS_JIEBA = False


def load_stopwords(path: str = "stopwords.txt") -> Set[str]:
    stop = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip()
                if w:
                    stop.add(w)
    except FileNotFoundError:
        # 返回空集合，调用方可以传入自定义停用词集
        pass
    return stop


def remove_special_expressions(text: str) -> str:
    """移除日期、时间、货币、百分比、邮件、url 等特殊表达式为一个空格。"""
    patterns = [
        # 日期：YYYY-MM-DD 或 YYYY/MM/DD 或 2023年5月6日
        r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?",
        r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}",
        # 时间 hh:mm(:ss)?
        r"\d{1,2}:\d{2}(?::\d{2})?",
        # 货币：￥ ¥ $ 元 欧元等
        r"[¥￥$€£]\s?\d+[\d,.]*",
        r"\d+[\d,.]*\s?(元|人民币|美金|美元|欧元)",
        # 百分比
        r"\d+(?:\.\d+)?%",
        # 邮箱
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        # url
        r"https?://[^\s]+",
        r"www\.[^\s]+",
    ]
    for p in patterns:
        text = re.sub(p, " ", text)
    return text


def remove_punctuation(text: str) -> str:
    """去掉标点和特殊字符，保留中文、英文数字、下划线和空白。"""
    # 保留汉字、英数字、下划线和空白
    return re.sub(r"[^\w\s\u4e00-\u9fff]", " ", text)


def tokenize(text: str) -> List[str]:
    if _HAS_JIEBA:
        return [t for t in jieba.lcut(text) if t.strip()]
    # fallback: 保留中文连续字符或英文/数字序列
    return re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+", text)


def compute_tf_df(docs_tokens: Iterable[List[str]]):
    tf = Counter()
    df = Counter()
    docs_tokens = list(docs_tokens)
    for tokens in docs_tokens:
        tf.update(tokens)
        df.update(set(tokens))
    return tf, df, len(docs_tokens)


def filter_by_freq_and_idf(
    docs_tokens: List[List[str]], high_pct: float = 0.03, rare_pct: float = 0.03
) -> List[List[str]]:
    """删除高频词（top high_pct）和非常罕见的词（IDF top rare_pct）。

    high_pct 和 rare_pct 都是相对于词汇表大小的比例（如 0.03 表示 top3%）。
    """
    tf, df, N = compute_tf_df(docs_tokens)
    if not tf:
        return docs_tokens

    vocab = list(tf.keys())
    vocab_size = len(vocab)

    # 高频词：按总出现次数排序，取 top high_pct
    top_k_high = max(1, math.ceil(vocab_size * high_pct))
    high_sorted = sorted(vocab, key=lambda w: tf[w], reverse=True)
    remove_high = set(high_sorted[:top_k_high])

    # IDF: log(N/(1+df))，罕见词 IDF 高，取 top rare_pct
    idf = {w: math.log(N / (1 + df[w])) for w in vocab}
    top_k_rare = max(1, math.ceil(vocab_size * rare_pct))
    rare_sorted = sorted(vocab, key=lambda w: idf[w], reverse=True)
    remove_rare = set(rare_sorted[:top_k_rare])

    removed = remove_high | remove_rare

    filtered = []
    for tokens in docs_tokens:
        filtered.append([t for t in tokens if t not in removed])
    return filtered


def remove_stopwords(tokens: List[str], stopwords: Set[str]) -> List[str]:
    if not stopwords:
        return tokens
    s = set(stopwords)
    return [t for t in tokens if t not in s]


def clean_corpus(
    docs: List[str], stopwords_path: str = "stopwords.txt", high_pct: float = 0.03, rare_pct: float = 0.03
) -> List[str]:
    """对一组文档进行完整清洗并返回清洗后的字符串列表（每个文档以空格分词连接）。

    参数：
    - docs: 文档字符串列表
    - stopwords_path: 停用词文件路径（utf-8），每行一个词。如果找不到文件会使用空停用词集。
    - high_pct: 删除高频词的比例（默认 0.03）
    - rare_pct: 删除罕见词（高IDF）的比例（默认 0.03）
    """
    stopwords = load_stopwords(stopwords_path)

    docs_tokens = []
    for doc in docs:
        if not isinstance(doc, str):
            doc = str(doc)
        t = remove_special_expressions(doc)
        t = remove_punctuation(t)
        toks = tokenize(t)
        toks = [w.strip() for w in toks if w.strip()]
        toks = remove_stopwords(toks, stopwords)
        docs_tokens.append(toks)

    docs_tokens = filter_by_freq_and_idf(docs_tokens, high_pct=high_pct, rare_pct=rare_pct)

    # 返回拼接好的字符串（以空格分开），方便后续向量化（如 Count/TF-IDF）
    return [" ".join(tokens) for tokens in docs_tokens]


if __name__ == "__main__":
    sample = [
        "2023年5月6日，售价为￥199.00，欢迎咨询：support@example.com",
        "这是一个测试文档，包含一些常见词和不常见词。",
    ]
    cleaned = clean_corpus(sample)
    print("原文:")
    for s in sample:
        print(" -", s)
    print("\n清洗后:")
    for s in cleaned:
        print(" -", s)
