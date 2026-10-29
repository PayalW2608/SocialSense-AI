"""
SocialSense AI — Agent Orchestrator
Routes user requests to the appropriate specialist agent and combines outputs.
Logs agent activity, results, and errors transparently.
"""
import time
from dataclasses import dataclass, field
from typing import Any
import pandas as pd

from agents.sentiment_agent import analyze_sentiment_text, analyze_sentiment_from_df
from agents.trend_agent import extract_hashtags, extract_keywords, get_topic_engagement, summarize_trends
from agents.competitor_agent import compute_engagement_metrics, get_sentiment_by_brand, analyze_competitors
from agents.campaign_agent import get_best_posting_times, get_platform_performance, generate_campaign_strategy
from agents.content_agent import generate_content, regenerate_single_caption


@dataclass
class AgentEvent:
    agent: str
    status: str          # "running" | "success" | "error"
    message: str
    duration_ms: float = 0.0
    result: Any = None


class AgentLog:
    """Simple in-memory activity log for agent events."""

    def __init__(self):
        self.events: list[AgentEvent] = []

    def add(self, event: AgentEvent):
        self.events.append(event)

    def clear(self):
        self.events.clear()


# Global log accessible by the UI
activity_log = AgentLog()


def _timed_call(agent_name: str, fn, *args, **kwargs):
    """Execute fn, record timing and errors to activity_log."""
    t0 = time.time()
    activity_log.add(AgentEvent(agent=agent_name, status="running", message="Starting…"))
    try:
        result = fn(*args, **kwargs)
        ms = round((time.time() - t0) * 1000)
        activity_log.add(AgentEvent(
            agent=agent_name, status="success",
            message=f"Completed in {ms} ms", duration_ms=ms, result=result,
        ))
        return result, None
    except Exception as e:
        ms = round((time.time() - t0) * 1000)
        activity_log.add(AgentEvent(
            agent=agent_name, status="error",
            message=str(e), duration_ms=ms,
        ))
        return None, str(e)


# ── Public orchestration functions ────────────────────────────────────────────

def run_sentiment_pipeline(df: pd.DataFrame, raw_text: str = "") -> dict:
    """Run the full sentiment analysis pipeline."""
    activity_log.clear()
    result = {}

    dist, err = _timed_call("Sentiment Agent", analyze_sentiment_from_df, df)
    if dist:
        result["distribution"] = dist

    if raw_text.strip():
        summary, err2 = _timed_call("Sentiment Agent (AI)", analyze_sentiment_text, raw_text)
        if summary:
            result["ai_summary"] = summary

    return result


def run_trend_pipeline(df: pd.DataFrame) -> dict:
    """Run the full trend discovery pipeline."""
    activity_log.clear()

    hashtag_df, _ = _timed_call("Trend Agent — Hashtags", extract_hashtags, df)
    keyword_df, _ = _timed_call("Trend Agent — Keywords", extract_keywords, df)
    topic_df, _ = _timed_call("Trend Agent — Topics", get_topic_engagement, df)

    hashtag_df = hashtag_df if hashtag_df is not None else pd.DataFrame()
    keyword_df = keyword_df if keyword_df is not None else pd.DataFrame()
    topic_df = topic_df if topic_df is not None else pd.DataFrame()

    summary, _ = _timed_call(
        "Trend Agent — AI Summary", summarize_trends,
        hashtag_df, keyword_df, topic_df,
    )

    return {
        "hashtags": hashtag_df,
        "keywords": keyword_df,
        "topics": topic_df,
        "ai_summary": summary or "Analysis unavailable.",
    }


def run_competitor_pipeline(competitor_df: pd.DataFrame) -> dict:
    """Run the competitor intelligence pipeline."""
    activity_log.clear()

    metrics, _ = _timed_call(
        "Competitor Agent — Metrics", compute_engagement_metrics, competitor_df
    )
    sentiment, _ = _timed_call(
        "Competitor Agent — Sentiment", get_sentiment_by_brand, competitor_df
    )
    report, _ = _timed_call(
        "Competitor Agent — AI Report", analyze_competitors, competitor_df
    )

    return {
        "metrics": metrics if metrics is not None else pd.DataFrame(),
        "sentiment": sentiment if sentiment is not None else pd.DataFrame(),
        "ai_report": report or "Analysis unavailable.",
    }


def run_campaign_pipeline(
    df: pd.DataFrame,
    brand: str,
    industry: str,
    objective: str,
    target_audience: str,
) -> dict:
    """Run the campaign strategy pipeline."""
    activity_log.clear()

    timing_df, _ = _timed_call("Campaign Agent — Posting Times", get_best_posting_times, df)
    platform_df, _ = _timed_call("Campaign Agent — Platforms", get_platform_performance, df)

    # Derive analytics summaries for the AI prompt
    from agents.trend_agent import extract_hashtags as _ht
    hashtag_df, _ = _timed_call("Campaign Agent — Hashtags", _ht, df)

    top_hashtags = (
        ", ".join(hashtag_df["hashtag"].head(8).tolist())
        if hashtag_df is not None and not hashtag_df.empty
        else "Not available"
    )
    top_topics = (
        ", ".join(df["topic"].value_counts().head(5).index.tolist())
        if "topic" in df.columns
        else "Not available"
    )
    best_platforms = (
        ", ".join(platform_df["platform"].head(3).tolist())
        if platform_df is not None and not platform_df.empty
        else "Not available"
    )

    from agents.sentiment_agent import analyze_sentiment_from_df as _sent
    dist = _sent(df)
    sentiment_summary = (
        f"{dist.get('positive_pct', 0)}% positive, "
        f"{dist.get('negative_pct', 0)}% negative, "
        f"{dist.get('neutral_pct', 0)}% neutral"
        if "error" not in dist
        else "Not available"
    )

    strategy, _ = _timed_call(
        "Campaign Agent — AI Strategy",
        generate_campaign_strategy,
        brand, industry, objective, target_audience,
        top_hashtags, top_topics, best_platforms, sentiment_summary,
    )

    return {
        "posting_times": timing_df if timing_df is not None else pd.DataFrame(),
        "platform_performance": platform_df if platform_df is not None else pd.DataFrame(),
        "ai_strategy": strategy or "Strategy generation unavailable.",
    }


def run_content_pipeline(
    brand: str, industry: str, target_audience: str,
    platform: str, tone: str, topic: str, objective: str,
    extra_context: str = "",
) -> dict:
    """Run the AI Content Studio pipeline."""
    activity_log.clear()

    content, err = _timed_call(
        "Content Agent", generate_content,
        brand, industry, target_audience, platform, tone, topic, objective, extra_context,
    )
    return {"content": content or f"Content generation failed: {err}"}
