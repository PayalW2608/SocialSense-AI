"""
SocialSense AI — AI Content Studio Agent
Uses IBM Granite to generate captions, hashtags, CTAs, and campaign ideas.
"""
from config import generate_text


PLATFORM_GUIDANCE = {
    "Instagram": "visually descriptive, emoji-friendly, 2200-char limit, 3–5 hashtags inline",
    "Facebook": "conversational, longer-form allowed, community-oriented, 1–2 hashtags",
    "LinkedIn": "professional, thought-leadership tone, data-backed, 3 hashtags max",
    "X (Twitter)": "concise, punchy, under 280 chars per tweet, 1–2 hashtags, strong hook",
}

TONE_EXAMPLES = {
    "Professional": "authoritative, data-driven, credible",
    "Casual & Fun": "friendly, relatable, light humour",
    "Inspirational": "motivating, aspirational, empowering",
    "Educational": "informative, clear, step-by-step",
    "Urgent / Promotional": "FOMO-driven, time-sensitive, strong CTA",
}


def generate_content(
    brand: str,
    industry: str,
    target_audience: str,
    platform: str,
    tone: str,
    topic: str,
    objective: str,
    extra_context: str = "",
) -> str:
    """
    Generate platform-specific social media content using IBM Granite.
    Returns structured content: captions, hashtags, CTAs, and campaign ideas.
    """
    platform_note = PLATFORM_GUIDANCE.get(platform, "general social media best practices")
    tone_note = TONE_EXAMPLES.get(tone, tone)

    prompt = f"""You are an expert social media content creator specialising in brand marketing.

Generate compelling social media content for the following brief:

Brand: {brand}
Industry: {industry}
Target Audience: {target_audience}
Platform: {platform} ({platform_note})
Tone: {tone} — {tone_note}
Topic/Theme: {topic}
Campaign Objective: {objective}
{f"Additional Context: {extra_context}" if extra_context else ""}

Generate ALL of the following:

## 📝 Caption Option 1 (Short & Punchy)
[Write a short, high-impact caption optimised for {platform}]

## 📝 Caption Option 2 (Story-Driven)
[Write a narrative-style caption that builds connection]

## 📝 Caption Option 3 (Educational / Value-Add)
[Write a caption that teaches or informs the audience]

## #️⃣ Hashtag Set (15 hashtags)
[Provide 5 high-volume + 5 mid-range + 5 niche hashtags relevant to {industry} and {topic}]

## 🎯 Calls to Action (5 variations)
[Write 5 distinct CTAs suitable for {objective}]

## 💡 Campaign Ideas (3 ideas)
[Suggest 3 creative campaign concepts with title, description, and expected outcome]

## 📅 Content Series Suggestion
[Suggest a 5-post content series idea around this theme]

Be creative, platform-native, and audience-aware. Avoid clichés."""
    return generate_text(prompt, max_tokens=1200)


def regenerate_single_caption(
    brand: str,
    platform: str,
    tone: str,
    topic: str,
    feedback: str = "",
) -> str:
    """Regenerate a single caption incorporating user feedback."""
    prompt = f"""You are a creative social media copywriter.

Rewrite a fresh {tone} caption for {brand} on {platform} about "{topic}".
{f"User feedback on previous version: {feedback}" if feedback else "Make it distinctly different from a typical generic post."}

Platform constraints: {PLATFORM_GUIDANCE.get(platform, "general best practices")}

Provide ONE polished caption ready to post, followed by 5 relevant hashtags."""
    return generate_text(prompt, max_tokens=300)
