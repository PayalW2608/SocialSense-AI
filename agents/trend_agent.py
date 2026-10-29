"""
SocialSense AI — Trend Discovery Agent
Identifies keywords, hashtags, and emerging topics from post data.
"""
import re
from collections import Counter

import pandas as pd
from config import generate_text


# ── Keyword / hashtag extraction ──────────────────────────────────────────────

def extract_hashtags(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of (hashtag, count) sorted by frequency."""
    if "hashtags" not in df.columns and "post_text" not in df.columns:
        return pd.DataFrame(columns=["hashtag", "count"])
    text_col = "hashtags" if "hashtags" in df.columns else "post_text"
    all_tags: list[str] = []
    for val in df[text_col].dropna():
        all_tags.extend(re.findall(r"#\w+", str(val).lower()))
    counts = Counter(all_tags).most_common(20)
    return pd.DataFrame(counts, columns=["hashtag", "count"])


def extract_keywords(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Return the most frequent non-stopword words from post_text."""
    if "post_text" not in df.columns:
        return pd.DataFrame(columns=["keyword", "count"])
    stop = {
        "the","a","an","and","or","but","in","on","at","to","for","of",
        "with","is","it","our","we","us","you","your","this","that","are",
        "be","was","were","have","has","do","does","not","from","by","as",
        "i","my","me","their","they","he","she","his","her","its","will",
        "can","more","so","all","about","just","what","how","new","get",
        "let","know","join","latest","exciting","excited",
    }
    words: list[str] = []
    for text in df["post_text"].dropna():
        tokens = re.findall(r"\b[a-z]{4,}\b", str(text).lower())
        words.extend([t for t in tokens if t not in stop])
    counts = Counter(words).most_common(top_n)
    return pd.DataFrame(counts, columns=["keyword", "count"])


def get_topic_engagement(df: pd.DataFrame) -> pd.DataFrame:
    """Average engagement per topic."""
    if "topic" not in df.columns or "engagement" not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby("topic")["engagement"]
        .mean()
        .reset_index()
        .rename(columns={"engagement": "avg_engagement"})
        .sort_values("avg_engagement", ascending=False)
    )


# ── AI Trend Summary ──────────────────────────────────────────────────────────

def summarize_trends(hashtag_df: pd.DataFrame, keyword_df: pd.DataFrame,
                     topic_df: pd.DataFrame) -> str:
    """Ask IBM Granite to summarise discovered trends and give recommendations."""
    top_tags = ", ".join(hashtag_df["hashtag"].head(10).tolist()) if not hashtag_df.empty else "N/A"
    top_kws = ", ".join(keyword_df["keyword"].head(10).tolist()) if not keyword_df.empty else "N/A"
    top_topics = (
        ", ".join(topic_df["topic"].head(5).tolist()) if not topic_df.empty else "N/A"
    )

    prompt = f"""You are a social media trend analyst.

Based on the following data from social media posts, provide a comprehensive trend report:

Top Hashtags: {top_tags}
Top Keywords: {top_kws}
Highest-Engagement Topics: {top_topics}

Your report must include:
1. **Emerging Trends** — Describe 3 key trends you see in the data.
2. **Audience Interest** — What topics are resonating most and why?
3. **Content Opportunities** — Suggest 3 content themes a brand should explore now.
4. **Hashtag Strategy** — Which hashtags should be prioritised and why?
5. **Actionable Recommendations** — 3 specific next steps for the social media team.

Be evidence-based, reference the actual hashtags and keywords, and keep each section concise."""
    return generate_text(prompt, max_tokens=750)
