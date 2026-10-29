"""
SocialSense AI — Competitor Intelligence Agent
Compares competitor metrics, hashtags, content themes, and sentiment.
Engagement rate formula: (likes + comments + shares) / followers * 100
(only calculated when follower counts are available).
"""
import pandas as pd
import numpy as np
from config import generate_text


def compute_engagement_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-brand engagement metrics.
    Accepts either:
      - a raw posts DataFrame with columns: brand/platform, likes, comments, shares
      - a pre-aggregated benchmark DataFrame with columns: brand, avg_likes, avg_comments, avg_shares
    """
    group_col = "brand" if "brand" in df.columns else "platform"
    if group_col not in df.columns:
        return pd.DataFrame()

    # Pre-aggregated benchmark format (avg_likes already present)
    if "avg_likes" in df.columns:
        result = df[[group_col]].copy()
        result["avg_likes"] = df.get("avg_likes", 0)
        result["avg_comments"] = df.get("avg_comments", 0)
        result["avg_shares"] = df.get("avg_shares", 0)
        result["total_posts"] = df.get("total_posts", 0)
        result["avg_engagement_score"] = (
            result["avg_likes"] + result["avg_comments"] * 2 + result["avg_shares"] * 3
        ).round(1)
        return result

    # Raw posts format — aggregate on the fly
    if not all(c in df.columns for c in ["likes", "comments", "shares"]):
        return pd.DataFrame()

    agg = (
        df.groupby(group_col)
        .agg(
            avg_likes=("likes", "mean"),
            avg_comments=("comments", "mean"),
            avg_shares=("shares", "mean"),
            total_posts=(group_col, "count"),
        )
        .reset_index()
    )
    agg["avg_engagement_score"] = (
        agg["avg_likes"] + agg["avg_comments"] * 2 + agg["avg_shares"] * 3
    ).round(1)
    return agg


def get_sentiment_by_brand(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot sentiment counts per brand."""
    group_col = "brand" if "brand" in df.columns else "platform"
    if group_col not in df.columns or "sentiment" not in df.columns:
        return pd.DataFrame()
    pivot = (
        df.groupby([group_col, "sentiment"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    pivot.columns.name = None
    return pivot


def analyze_competitors(competitor_df: pd.DataFrame) -> str:
    """
    Ask IBM Granite for a competitor intelligence report based on benchmark data.
    competitor_df must have columns: brand, avg_likes, avg_comments, avg_shares,
    total_posts, positive_pct, negative_pct, top_hashtag.
    """
    if competitor_df.empty:
        return "No competitor data available for analysis."

    table_str = competitor_df.to_string(index=False)

    prompt = f"""You are an expert social media competitive intelligence analyst.

Analyse the following competitor benchmark data and generate a strategic report:

{table_str}

Your report must cover:
1. **Performance Comparison** — Rank competitors by engagement and explain what drives top performance.
2. **Content Strategy Insights** — What content patterns are working best for leading brands?
3. **Hashtag Opportunities** — Which hashtags offer the highest potential reach?
4. **Sentiment Landscape** — Compare brand perception across competitors.
5. **Gaps & Opportunities** — Where can YourBrand improve or differentiate?
6. **3 Immediate Recommendations** — Specific, actionable steps to outperform competitors.

Be specific, cite the data provided, and avoid generic advice."""
    return generate_text(prompt, max_tokens=800)
