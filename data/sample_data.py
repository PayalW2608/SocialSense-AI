"""
SocialSense AI — Illustrative sample data used when no real data is imported.
All values are fictional and clearly marked as sample/illustrative.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

SAMPLE_LABEL = "⚠️ Sample / Illustrative Data — Not real social media data"

def get_sample_posts(n: int = 120) -> pd.DataFrame:
    """Generate n rows of illustrative social media post data."""
    np.random.seed(42)
    platforms = ["Instagram", "Facebook", "LinkedIn", "X (Twitter)"]
    sentiments = ["positive", "negative", "neutral"]
    weights_sent = [0.55, 0.20, 0.25]
    topics = [
        "product launch", "customer support", "brand awareness",
        "sustainability", "innovation", "community", "sales", "events",
    ]
    hashtags_pool = [
        "#innovation", "#sustainability", "#brandlove", "#trending",
        "#tech", "#community", "#growth", "#marketing", "#AI", "#startup",
    ]

    start_date = datetime.now() - timedelta(days=60)
    dates = [start_date + timedelta(hours=i * 12) for i in range(n)]

    rows = []
    for i, date in enumerate(dates):
        platform = np.random.choice(platforms)
        sentiment = np.random.choice(sentiments, p=weights_sent)
        likes = int(np.random.exponential(200))
        comments = int(np.random.exponential(30))
        shares = int(np.random.exponential(15))
        engagement = likes + comments * 2 + shares * 3
        hashtags = " ".join(np.random.choice(hashtags_pool, size=np.random.randint(2, 5), replace=False))
        topic = np.random.choice(topics)
        post_text = (
            f"Excited about our latest {topic}! {hashtags} "
            f"Join the conversation and let us know your thoughts."
        )
        rows.append({
            "date": date.strftime("%Y-%m-%d %H:%M"),
            "platform": platform,
            "post_text": post_text,
            "hashtags": hashtags,
            "topic": topic,
            "sentiment": sentiment,
            "likes": likes,
            "comments": comments,
            "shares": shares,
            "engagement": engagement,
        })
    return pd.DataFrame(rows)


def get_sample_competitors() -> pd.DataFrame:
    """Return illustrative competitor benchmark data."""
    np.random.seed(7)
    competitors = ["BrandAlpha", "BrandBeta", "BrandGamma", "YourBrand"]
    data = []
    for brand in competitors:
        data.append({
            "brand": brand,
            "avg_likes": int(np.random.uniform(150, 800)),
            "avg_comments": int(np.random.uniform(20, 120)),
            "avg_shares": int(np.random.uniform(10, 60)),
            "total_posts": int(np.random.uniform(30, 200)),
            "positive_pct": round(np.random.uniform(45, 80), 1),
            "negative_pct": round(np.random.uniform(5, 25), 1),
            "top_hashtag": np.random.choice(["#innovation", "#brandlove", "#tech", "#AI"]),
        })
    return pd.DataFrame(data)


SAMPLE_CSV_CONTENT = """date,platform,post_text,hashtags,topic,sentiment,likes,comments,shares
2024-05-01 09:00,Instagram,Loving our new collection! #fashion #style,#fashion #style,product launch,positive,450,32,18
2024-05-01 14:00,Facebook,Customer support is our priority. #help #support,#help #support,customer support,neutral,120,45,8
2024-05-02 10:00,LinkedIn,Proud to announce our sustainability goals! #green,#green,sustainability,positive,890,67,44
2024-05-02 16:00,X (Twitter),Experiencing delays — sorry for inconvenience. #issue,#issue,customer support,negative,30,88,5
2024-05-03 11:00,Instagram,Join our community event! #community #event,#community #event,events,positive,310,29,21
"""
