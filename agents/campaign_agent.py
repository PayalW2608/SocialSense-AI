"""
SocialSense AI — Campaign Strategy Agent
Recommends content themes, posting schedules, hashtags, and campaign ideas
based on available analytics data.
"""
import pandas as pd
from config import generate_text


def get_best_posting_times(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify hours and days with the highest average engagement.
    Requires at least 20 rows with 'date' and 'engagement' columns.
    Returns a DataFrame with hour and avg_engagement (or empty if insufficient data).
    """
    MIN_ROWS = 20
    if "date" not in df.columns or "engagement" not in df.columns or len(df) < MIN_ROWS:
        return pd.DataFrame()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.day_name()

    hourly = (
        df.groupby("hour")["engagement"]
        .mean()
        .reset_index()
        .rename(columns={"engagement": "avg_engagement"})
        .sort_values("avg_engagement", ascending=False)
    )
    return hourly


def get_platform_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Average engagement and post count per platform."""
    if "platform" not in df.columns or "engagement" not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby("platform")
        .agg(avg_engagement=("engagement", "mean"), post_count=("platform", "count"))
        .reset_index()
        .sort_values("avg_engagement", ascending=False)
    )


def generate_campaign_strategy(
    brand: str,
    industry: str,
    objective: str,
    target_audience: str,
    top_hashtags: str,
    top_topics: str,
    best_platforms: str,
    sentiment_summary: str,
) -> str:
    """Ask IBM Granite to produce a full campaign strategy."""
    prompt = f"""You are an expert social media strategist.

Generate a comprehensive, data-driven campaign strategy for the following brand:

Brand: {brand}
Industry: {industry}
Campaign Objective: {objective}
Target Audience: {target_audience}
Top Performing Hashtags: {top_hashtags}
Top Engaging Topics: {top_topics}
Best Performing Platforms: {best_platforms}
Current Sentiment: {sentiment_summary}

Your strategy must include:
1. **Campaign Theme & Narrative** — A compelling overarching theme with a tagline.
2. **Content Pillars** — 4 distinct content pillars with example post ideas for each.
3. **Hashtag Strategy** — Primary, secondary, and branded hashtags with rationale.
4. **Target Audience Personas** — 2 detailed audience personas with pain points and motivations.
5. **Platform-Specific Tactics** — Tailored approach for Instagram, LinkedIn, Facebook, and X.
6. **Posting Schedule** — Recommended frequency and timing per platform.
7. **KPIs & Success Metrics** — What to measure and target benchmarks.
8. **Risk Mitigation** — Potential pitfalls and how to avoid them.

Base recommendations on the analytics data provided. Explain the reasoning behind each recommendation."""
    return generate_text(prompt, max_tokens=1200)
