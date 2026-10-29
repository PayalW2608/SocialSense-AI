"""
SocialSense AI — Flask Application Entry Point
Replaces the Streamlit version; all agent logic is unchanged.
Run with:  python app.py
"""
import io
import json
import os
import re
import base64
from collections import Counter

import pandas as pd
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, jsonify, send_file, flash,
)
from werkzeug.utils import secure_filename

from config import generate_text
from data.sample_data import get_sample_posts, get_sample_competitors, SAMPLE_CSV_CONTENT
from agents.orchestrator import (
    run_sentiment_pipeline, run_trend_pipeline,
    run_competitor_pipeline, run_campaign_pipeline,
    run_content_pipeline, activity_log,
)
from agents.sentiment_agent import get_sentiment_over_time
from agents.trend_agent import extract_hashtags, extract_keywords, get_topic_engagement
from agents.competitor_agent import compute_engagement_metrics
from agents.campaign_agent import get_best_posting_times, get_platform_performance
from agents.content_agent import regenerate_single_caption, PLATFORM_GUIDANCE

app = Flask(__name__)
app.secret_key = os.urandom(24)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Session helpers ────────────────────────────────────────────────────────────

def _get_df() -> pd.DataFrame:
    """Return the active DataFrame from session, or the sample."""
    if "csv_data" in session:
        try:
            return pd.read_json(io.StringIO(session["csv_data"]), orient="split")
        except Exception:
            pass
    return get_sample_posts()


def _store_df(df: pd.DataFrame):
    session["csv_data"] = df.to_json(orient="split", date_format="iso")
    session["using_sample"] = False


def _df_to_chart(df: pd.DataFrame, x: str, y: str, kind: str = "bar",
                 color: str = None, labels: dict = None) -> dict:
    """Convert a DataFrame slice to a Plotly-compatible JSON dict."""
    if df.empty:
        return {}
    data = {"x": df[x].tolist(), "y": df[y].tolist(), "type": kind}
    if color and color in df.columns:
        data["marker"] = {"color": df[color].tolist()}
    if labels:
        data["name"] = labels.get(y, y)
    return data


def _sentiment_pie(dist: dict) -> str:
    """Return a Plotly pie chart JSON string for sentiment distribution."""
    import json
    labels = ["positive", "negative", "neutral"]
    values = [dist.get(l, 0) for l in labels]
    colors = ["#22c55e", "#ef4444", "#94a3b8"]
    fig = {
        "data": [{"type": "pie", "labels": labels, "values": values,
                  "hole": 0.4,
                  "marker": {"colors": colors},
                  "textinfo": "percent+label"}],
        "layout": {"margin": {"t": 20, "b": 20, "l": 20, "r": 20},
                   "showlegend": True, "height": 300,
                   "paper_bgcolor": "transparent",
                   "plot_bgcolor": "transparent"},
    }
    return json.dumps(fig)


def _bar_chart(x_vals, y_vals, title="", color="#3b82f6",
               orientation="v", height=300) -> str:
    import json
    trace = {
        "type": "bar",
        "x": x_vals if orientation == "v" else y_vals,
        "y": y_vals if orientation == "v" else x_vals,
        "orientation": orientation,
        "marker": {"color": color},
    }
    fig = {
        "data": [trace],
        "layout": {
            "title": title,
            "height": height,
            "margin": {"t": 30, "b": 60, "l": 120, "r": 20},
            "paper_bgcolor": "transparent",
            "plot_bgcolor": "transparent",
            "yaxis": {"autorange": "reversed"} if orientation == "h" else {},
        },
    }
    return json.dumps(fig)


def _line_chart(x_vals, y_vals, title="", color="#3b82f6") -> str:
    import json
    fig = {
        "data": [{"type": "scatter", "mode": "lines+markers",
                  "x": [str(v) for v in x_vals], "y": y_vals,
                  "line": {"color": color}}],
        "layout": {
            "title": title,
            "height": 280,
            "margin": {"t": 30, "b": 50, "l": 60, "r": 20},
            "paper_bgcolor": "transparent",
            "plot_bgcolor": "transparent",
        },
    }
    return json.dumps(fig)


def _log_events():
    return [{"agent": e.agent, "status": e.status, "message": e.message,
             "ms": e.duration_ms} for e in activity_log.events]


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return redirect(url_for("dashboard"))


# ── Dashboard ──────────────────────────────────────────────────────────────────

@app.route("/dashboard")
def dashboard():
    df = _get_df()
    using_sample = session.get("using_sample", True)

    total_posts = len(df)
    total_eng   = int(df["engagement"].sum()) if "engagement" in df.columns else 0
    avg_eng     = round(df["engagement"].mean(), 1) if "engagement" in df.columns else 0
    pos_pct     = 0
    if "sentiment" in df.columns and total_posts:
        pos_pct = round(df[df["sentiment"] == "positive"].shape[0] / total_posts * 100, 1)

    # Sentiment pie
    dist = {}
    if "sentiment" in df.columns:
        vc = df["sentiment"].value_counts()
        dist = {l: int(vc.get(l, 0)) for l in ["positive", "negative", "neutral"]}
        dist["total"] = total_posts
    sentiment_chart = _sentiment_pie(dist) if dist else ""

    # Platform bar
    platform_chart = ""
    if "platform" in df.columns:
        pc = df["platform"].value_counts()
        platform_chart = _bar_chart(pc.index.tolist(), pc.values.tolist(),
                                    color="#6366f1", height=280)

    # Engagement over time
    engagement_chart = ""
    if "date" in df.columns and "engagement" in df.columns:
        ts = df.copy()
        ts["date"] = pd.to_datetime(ts["date"], errors="coerce")
        ts = ts.dropna(subset=["date"])
        ts["date_only"] = ts["date"].dt.date
        grp = ts.groupby("date_only")["engagement"].sum().reset_index()
        engagement_chart = _line_chart(
            grp["date_only"].tolist(), grp["engagement"].tolist(), color="#3b82f6"
        )

    # Top hashtags
    hashtag_chart = ""
    if "hashtags" in df.columns:
        tags = []
        for v in df["hashtags"].dropna():
            tags.extend(re.findall(r"#\w+", str(v).lower()))
        top = Counter(tags).most_common(10)
        if top:
            ht_df = pd.DataFrame(top, columns=["hashtag", "count"])
            hashtag_chart = _bar_chart(
                ht_df["count"].tolist(), ht_df["hashtag"].tolist(),
                orientation="h", color="#0ea5e9", height=320
            )

    # Top topics
    topic_chart = ""
    if "topic" in df.columns and "engagement" in df.columns:
        te = df.groupby("topic")["engagement"].mean().sort_values(ascending=False).head(8)
        topic_chart = _bar_chart(
            te.values.tolist(), te.index.tolist(),
            orientation="h", color="#22c55e", height=320
        )

    return render_template(
        "dashboard.html",
        using_sample=using_sample,
        total_posts=total_posts,
        total_eng=total_eng,
        avg_eng=avg_eng,
        pos_pct=pos_pct,
        sentiment_chart=sentiment_chart,
        platform_chart=platform_chart,
        engagement_chart=engagement_chart,
        hashtag_chart=hashtag_chart,
        topic_chart=topic_chart,
    )


# ── Data Import ────────────────────────────────────────────────────────────────

@app.route("/data-import", methods=["GET", "POST"])
def data_import():
    message = None
    error   = None

    if request.method == "POST":
        action = request.form.get("action", "")

        if action == "use_sample":
            session.pop("csv_data", None)
            session["using_sample"] = True
            message = f"Loaded {len(get_sample_posts())} illustrative sample posts."

        elif action == "upload_csv":
            f = request.files.get("csv_file")
            if not f or f.filename == "":
                error = "No file selected."
            else:
                try:
                    df = pd.read_csv(f)
                    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
                    if "engagement" not in df.columns:
                        if all(c in df.columns for c in ["likes", "comments", "shares"]):
                            df["engagement"] = (df["likes"]
                                                + df["comments"] * 2
                                                + df["shares"] * 3)
                    if "sentiment" in df.columns:
                        df["sentiment"] = df["sentiment"].str.lower().str.strip()
                    _store_df(df)
                    message = f"Uploaded {len(df)} posts successfully."
                except Exception as e:
                    error = f"Error reading CSV: {e}"

        return redirect(url_for("data_import",
                                message=message or "",
                                error=error or ""))

    # GET
    message = request.args.get("message", "")
    error   = request.args.get("error", "")
    return render_template(
        "data_import.html",
        sample_csv=SAMPLE_CSV_CONTENT,
        message=message, error=error,
    )


@app.route("/download-sample-csv")
def download_sample_csv():
    return send_file(
        io.BytesIO(SAMPLE_CSV_CONTENT.encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="socialsense_sample.csv",
    )


# ── AI Content Studio ──────────────────────────────────────────────────────────

@app.route("/content-studio", methods=["GET", "POST"])
def content_studio():
    result_content = None
    form_data = {}
    error = None

    if request.method == "POST":
        action = request.form.get("action", "generate")
        form_data = {k: v for k, v in request.form.items()}

        if action == "generate":
            brand          = request.form.get("brand", "").strip()
            industry       = request.form.get("industry", "").strip()
            target_audience= request.form.get("target_audience", "").strip()
            platform       = request.form.get("platform", "Instagram")
            tone           = request.form.get("tone", "Professional")
            topic          = request.form.get("topic", "").strip()
            objective      = request.form.get("objective", "Brand Awareness")
            extra_context  = request.form.get("extra_context", "").strip()

            if not brand or not topic:
                error = "Please enter at least a Brand Name and Topic."
            else:
                res = run_content_pipeline(
                    brand=brand, industry=industry,
                    target_audience=target_audience,
                    platform=platform, tone=tone,
                    topic=topic, objective=objective,
                    extra_context=extra_context,
                )
                result_content = res.get("content", "")
                session["last_content"] = result_content
                session["last_content_meta"] = {
                    "brand": brand, "platform": platform,
                    "tone": tone, "topic": topic,
                }

        elif action == "regenerate":
            meta = session.get("last_content_meta", {})
            feedback = request.form.get("feedback", "")
            result_content = regenerate_single_caption(
                brand=meta.get("brand", ""),
                platform=meta.get("platform", "Instagram"),
                tone=meta.get("tone", "Professional"),
                topic=meta.get("topic", ""),
                feedback=feedback,
            )

    return render_template(
        "content_studio.html",
        platforms=list(PLATFORM_GUIDANCE.keys()),
        tones=["Professional", "Casual & Fun", "Inspirational",
               "Educational", "Urgent / Promotional"],
        objectives=["Brand Awareness", "Lead Generation", "Product Launch",
                    "Community Engagement", "Event Promotion",
                    "Customer Retention", "Thought Leadership", "Sales Conversion"],
        result_content=result_content,
        form_data=form_data,
        error=error,
        log=_log_events(),
        last_meta=session.get("last_content_meta"),
    )


# ── Sentiment Analysis ────────────────────────────────────────────────────────

@app.route("/sentiment", methods=["GET", "POST"])
def sentiment():
    df = _get_df()
    using_sample = session.get("using_sample", True)

    ai_summary   = None
    raw_text     = ""
    error        = None

    if request.method == "POST":
        raw_text = request.form.get("raw_text", "").strip()
        if not raw_text:
            error = "Please enter some text to analyse."
        else:
            result = run_sentiment_pipeline(df, raw_text=raw_text)
            ai_summary = result.get("ai_summary", "")

    # Always compute numeric distribution from df
    result_df = run_sentiment_pipeline(df)
    dist = result_df.get("distribution", {})

    # Sentiment pie
    sent_chart = _sentiment_pie(dist) if dist and "error" not in dist else ""

    # Sentiment over time
    time_chart = ""
    ts_df = get_sentiment_over_time(df)
    if not ts_df.empty:
        import json
        traces = []
        colors = {"positive": "#22c55e", "negative": "#ef4444", "neutral": "#94a3b8"}
        for col in ["positive", "negative", "neutral"]:
            if col in ts_df.columns:
                traces.append({
                    "type": "scatter", "mode": "lines",
                    "name": col,
                    "x": [str(d) for d in ts_df["date"].tolist()],
                    "y": ts_df[col].tolist(),
                    "line": {"color": colors[col]},
                    "fill": "tozeroy",
                    "fillcolor": colors[col].replace("#", "rgba(").rstrip(")") + ",0.15)",
                })
        time_chart = json.dumps({
            "data": traces,
            "layout": {
                "height": 280,
                "margin": {"t": 20, "b": 50, "l": 50, "r": 20},
                "paper_bgcolor": "transparent",
                "plot_bgcolor": "transparent",
            },
        })

    # Platform sentiment grouped bar
    plat_chart = ""
    if "platform" in df.columns and "sentiment" in df.columns:
        import json
        grp = df.groupby(["platform", "sentiment"]).size().reset_index(name="count")
        colors = {"positive": "#22c55e", "negative": "#ef4444", "neutral": "#94a3b8"}
        traces = []
        for sent in ["positive", "negative", "neutral"]:
            sub = grp[grp["sentiment"] == sent]
            traces.append({
                "type": "bar", "name": sent,
                "x": sub["platform"].tolist(),
                "y": sub["count"].tolist(),
                "marker": {"color": colors[sent]},
            })
        plat_chart = json.dumps({
            "data": traces,
            "layout": {
                "barmode": "group", "height": 300,
                "margin": {"t": 20, "b": 60, "l": 50, "r": 20},
                "paper_bgcolor": "transparent",
                "plot_bgcolor": "transparent",
            },
        })

    return render_template(
        "sentiment.html",
        using_sample=using_sample,
        dist=dist,
        sent_chart=sent_chart,
        time_chart=time_chart,
        plat_chart=plat_chart,
        ai_summary=ai_summary,
        raw_text=raw_text,
        error=error,
        log=_log_events(),
    )


# ── Trend Discovery ────────────────────────────────────────────────────────────

@app.route("/trends")
def trends():
    df = _get_df()
    using_sample = session.get("using_sample", True)
    result = run_trend_pipeline(df)

    ht_df    = result["hashtags"]
    kw_df    = result["keywords"]
    topic_df = result["topics"]
    ai_summary = result["ai_summary"]

    ht_chart = ""
    if not ht_df.empty:
        ht_chart = _bar_chart(
            ht_df["count"].head(15).tolist(),
            ht_df["hashtag"].head(15).tolist(),
            orientation="h", color="#3b82f6", height=380
        )

    kw_chart = ""
    if not kw_df.empty:
        kw_chart = _bar_chart(
            kw_df["count"].head(15).tolist(),
            kw_df["keyword"].head(15).tolist(),
            orientation="h", color="#f97316", height=380
        )

    topic_chart = ""
    if not topic_df.empty:
        topic_chart = _bar_chart(
            topic_df["avg_engagement"].tolist(),
            topic_df["topic"].tolist(),
            orientation="h", color="#22c55e", height=320
        )

    # Volume over time
    vol_chart = ""
    if "date" in df.columns and "platform" in df.columns:
        import json
        ts = df.copy()
        ts["date"] = pd.to_datetime(ts["date"], errors="coerce")
        ts = ts.dropna(subset=["date"])
        ts["date_only"] = ts["date"].dt.date
        vol = ts.groupby(["date_only", "platform"]).size().reset_index(name="posts")
        pal = ["#3b82f6", "#f97316", "#22c55e", "#a855f7"]
        traces = []
        for i, plat in enumerate(vol["platform"].unique()):
            sub = vol[vol["platform"] == plat]
            traces.append({
                "type": "scatter", "mode": "lines+markers",
                "name": plat,
                "x": [str(d) for d in sub["date_only"].tolist()],
                "y": sub["posts"].tolist(),
                "line": {"color": pal[i % len(pal)]},
            })
        vol_chart = json.dumps({
            "data": traces,
            "layout": {
                "height": 300,
                "margin": {"t": 20, "b": 50, "l": 50, "r": 20},
                "paper_bgcolor": "transparent",
                "plot_bgcolor": "transparent",
            },
        })

    return render_template(
        "trends.html",
        using_sample=using_sample,
        ht_chart=ht_chart,
        kw_chart=kw_chart,
        topic_chart=topic_chart,
        vol_chart=vol_chart,
        ai_summary=ai_summary,
        log=_log_events(),
    )


# ── Competitor Intelligence ────────────────────────────────────────────────────

@app.route("/competitor", methods=["GET", "POST"])
def competitor():
    error = None
    comp_df = get_sample_competitors()
    using_sample_comp = True

    if request.method == "POST":
        f = request.files.get("comp_csv")
        if f and f.filename:
            try:
                comp_df = pd.read_csv(f)
                using_sample_comp = False
            except Exception as e:
                error = f"Error reading file: {e}"

    result = run_competitor_pipeline(comp_df)
    metrics_df = result["metrics"]
    ai_report  = result["ai_report"]

    # Grouped bar: avg_likes / comments / shares per brand
    eng_chart = ""
    if not comp_df.empty and "brand" in comp_df.columns:
        import json
        metrics_cols = [c for c in ["avg_likes", "avg_comments", "avg_shares"]
                        if c in comp_df.columns]
        pal = ["#3b82f6", "#f97316", "#22c55e"]
        traces = []
        for i, col in enumerate(metrics_cols):
            traces.append({
                "type": "bar", "name": col.replace("avg_", ""),
                "x": comp_df["brand"].tolist(),
                "y": comp_df[col].tolist(),
                "marker": {"color": pal[i]},
            })
        eng_chart = json.dumps({
            "data": traces,
            "layout": {
                "barmode": "group", "height": 320,
                "margin": {"t": 20, "b": 60, "l": 50, "r": 20},
                "paper_bgcolor": "transparent",
                "plot_bgcolor": "transparent",
            },
        })

    # Sentiment comparison
    sent_chart = ""
    if not comp_df.empty and all(c in comp_df.columns
                                  for c in ["brand", "positive_pct", "negative_pct"]):
        import json
        traces = [
            {"type": "bar", "name": "Positive %",
             "x": comp_df["brand"].tolist(),
             "y": comp_df["positive_pct"].tolist(),
             "marker": {"color": "#22c55e"}},
            {"type": "bar", "name": "Negative %",
             "x": comp_df["brand"].tolist(),
             "y": comp_df["negative_pct"].tolist(),
             "marker": {"color": "#ef4444"}},
        ]
        sent_chart = json.dumps({
            "data": traces,
            "layout": {
                "barmode": "group", "height": 300,
                "margin": {"t": 20, "b": 60, "l": 50, "r": 20},
                "paper_bgcolor": "transparent",
                "plot_bgcolor": "transparent",
            },
        })

    # Composite score bar
    score_chart = ""
    if not metrics_df.empty and "avg_engagement_score" in metrics_df.columns:
        brand_col = "brand" if "brand" in metrics_df.columns else "platform"
        score_chart = _bar_chart(
            metrics_df[brand_col].tolist(),
            metrics_df["avg_engagement_score"].tolist(),
            color="#a855f7", height=280
        )

    return render_template(
        "competitor.html",
        using_sample_comp=using_sample_comp,
        comp_data=comp_df.to_dict(orient="records"),
        eng_chart=eng_chart,
        sent_chart=sent_chart,
        score_chart=score_chart,
        ai_report=ai_report,
        error=error,
        log=_log_events(),
    )


# ── Campaign Strategy ─────────────────────────────────────────────────────────

@app.route("/campaign", methods=["GET", "POST"])
def campaign():
    df = _get_df()
    using_sample = session.get("using_sample", True)
    ai_strategy  = None
    timing_chart = ""
    platform_chart = ""
    error = None
    form_data = {}

    if request.method == "POST":
        brand           = request.form.get("brand", "").strip()
        industry        = request.form.get("industry", "").strip()
        target_audience = request.form.get("target_audience", "").strip()
        objective       = request.form.get("objective", "Brand Awareness")
        form_data = dict(request.form)

        if not brand:
            error = "Please enter a Brand Name."
        else:
            result = run_campaign_pipeline(
                df=df, brand=brand, industry=industry,
                objective=objective, target_audience=target_audience,
            )
            ai_strategy = result["ai_strategy"]

            timing_df   = result["posting_times"]
            platform_df = result["platform_performance"]

            if not timing_df.empty:
                timing_chart = _bar_chart(
                    timing_df["hour"].head(12).tolist(),
                    timing_df["avg_engagement"].head(12).tolist(),
                    color="#f59e0b", height=280
                )

            if not platform_df.empty:
                platform_chart = _bar_chart(
                    platform_df["platform"].tolist(),
                    platform_df["avg_engagement"].tolist(),
                    color="#06b6d4", height=280
                )

    return render_template(
        "campaign.html",
        using_sample=using_sample,
        objectives=["Brand Awareness", "Lead Generation", "Product Launch",
                    "Community Engagement", "Event Promotion",
                    "Customer Retention", "Thought Leadership", "Sales Conversion"],
        ai_strategy=ai_strategy,
        timing_chart=timing_chart,
        platform_chart=platform_chart,
        error=error,
        form_data=form_data,
        log=_log_events(),
    )


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000)
