"""
SocialSense AI — Sentiment Analysis Agent
Uses IBM Granite for language understanding; Pandas for numerical aggregation.
"""
import pandas as pd
from config import generate_text


def analyze_sentiment_from_df(df: pd.DataFrame) -> dict:
    """
    Compute sentiment distribution from a DataFrame that has a 'sentiment' column.
    Returns counts, percentages, and a summary dict.
    """
    if "sentiment" not in df.columns:
        return {"error": "No 'sentiment' column found in the uploaded data."}

    counts = df["sentiment"].value_counts()
    total = len(df)
    dist = {
        "positive": int(counts.get("positive", 0)),
        "negative": int(counts.get("negative", 0)),
        "neutral": int(counts.get("neutral", 0)),
        "total": total,
    }
    dist["positive_pct"] = round(dist["positive"] / total * 100, 1) if total else 0
    dist["negative_pct"] = round(dist["negative"] / total * 100, 1) if total else 0
    dist["neutral_pct"] = round(dist["neutral"] / total * 100, 1) if total else 0
    return dist


def analyze_sentiment_text(posts_text: str) -> str:
    """
    Ask IBM Granite to classify and summarise sentiment in free-form post text.
    """
    prompt = f"""You are a social media sentiment analysis expert.

Analyze the following social media posts and:
1. Classify the overall sentiment as Positive, Negative, or Neutral with a confidence level.
2. Identify the top 3 recurring emotional themes (e.g., excitement, frustration, trust).
3. Highlight any specific phrases that indicate sentiment shifts.
4. Provide a 3-sentence audience opinion summary.
5. Flag any potential reputation risks.

Posts:
\"\"\"
{posts_text[:3000]}
\"\"\"

Respond with clear headings for each section. Be concise and evidence-based."""
    return generate_text(prompt, max_tokens=700)


def get_sentiment_over_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sentiment counts by date for time-series plotting.
    Returns a DataFrame with columns: date, positive, negative, neutral.
    """
    if "date" not in df.columns or "sentiment" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["date_only"] = df["date"].dt.date
    pivot = (
        df.groupby(["date_only", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    pivot.columns.name = None
    for col in ["positive", "negative", "neutral"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot.rename(columns={"date_only": "date"})
    return pivot
