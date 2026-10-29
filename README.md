<div align="center">

# 📡 SocialSense AI

### Intelligent Social Media Analytics & Content Agent
**Powered by IBM Granite on watsonx.ai**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-black?logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![IBM watsonx.ai](https://img.shields.io/badge/IBM-watsonx.ai-054ADA?logo=ibm&logoColor=white)](https://www.ibm.com/watsonx)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Monitor conversations · Analyse sentiment · Discover trends · Generate content · Outpace competitors*

</div>

---

## 🌟 Overview

**SocialSense AI** is a fully agentic social media intelligence platform built with **Flask** and **IBM Granite** on **watsonx.ai**. It helps brands, influencers, and marketing teams:

- Monitor and understand audience sentiment across platforms
- Discover emerging trends and high-engagement content patterns
- Benchmark against competitors and identify strategic gaps
- Generate platform-ready captions, hashtags, and campaign plans using generative AI
- Make data-driven decisions with interactive Plotly dashboards

> ⚠️ **Data Notice**: Sample data is fictional and clearly labelled as illustrative.  
> Live social media API integration requires platform-specific credentials (see [Data Import](#-data-import--multimodal-input)).

---

## ✨ Features

| Module | Description |
|--------|-------------|
| 📊 **Overview Dashboard** | KPI cards (posts, engagement, sentiment), Plotly charts, platform breakdown, hashtag frequency, topic engagement |
| ✍️ **AI Content Studio** | IBM Granite–powered captions (3 styles), 15 hashtags, 5 CTAs, 3 campaign ideas, 5-post content series — Instagram, Facebook, LinkedIn & X |
| 💬 **Sentiment Analysis Agent** | Classifies sentiment (positive/negative/neutral), time-series trends, platform-level breakdown, free-text AI analysis |
| 🔥 **Trend Discovery Agent** | Keyword & hashtag frequency charts, topic engagement ranking, post-volume over time, AI-generated trend summary |
| 🏆 **Competitor Intelligence Agent** | Engagement benchmarking, composite scores, sentiment comparison, AI competitive intelligence report |
| 🎯 **Campaign Strategy Agent** | Best posting hours, platform performance, full AI campaign plan with personas, hashtag strategy, KPIs, and risk mitigation |
| 📥 **Data Import & Multimodal** | CSV upload with validation, sample CSV download, image upload, audio upload |

---

## 🏗️ Architecture

```
SocialSense AI/
├── app.py                        # Flask entry point — all routes & chart helpers
├── config.py                     # IBM Granite credentials & generate_text()
├── requirements.txt
│
├── templates/                    # Jinja2 HTML templates
│   ├── base.html                 # Shared layout: sidebar, CSS, Plotly JS CDN
│   ├── dashboard.html
│   ├── content_studio.html
│   ├── sentiment.html
│   ├── trends.html
│   ├── competitor.html
│   ├── campaign.html
│   └── data_import.html
│
├── agents/                       # AI Agent layer (no Flask dependency)
│   ├── orchestrator.py           # Routes requests, logs agent activity
│   ├── sentiment_agent.py        # Sentiment classification & summarisation
│   ├── trend_agent.py            # Hashtag/keyword extraction & trend summaries
│   ├── competitor_agent.py       # Competitor benchmarking & intelligence
│   ├── campaign_agent.py         # Posting schedules & campaign strategy
│   └── content_agent.py          # Caption, hashtag & campaign content generation
│
└── data/
    └── sample_data.py            # Illustrative sample data generator
```

### Agent Orchestration Flow

```
Browser Request (Flask route)
          │
          ▼
   ┌─────────────┐
   │ Orchestrator │  ← routes task, logs activity & timing
   └──────┬──────┘
          │ dispatches to
    ┌─────┼─────┬──────┬──────┐
    ▼     ▼     ▼      ▼      ▼
 Sent.  Trend  Comp  Camp  Content
 Agent  Agent  Agent Agent  Agent
    │     │     │      │      │
    └─────┴─────┴──────┴──────┘
                  │
                  ▼
          IBM Granite 3 8B
          (watsonx.ai API)
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip

### 1. Clone the repository

```bash
git clone https://github.com/PayalWankhade26/SocialSense-AI.git
cd SocialSense-AI
```

### 2. Create and activate a virtual environment (recommended)

```bash
# macOS / Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure IBM watsonx.ai credentials

The app ships with demo credentials pre-configured in [`config.py`](config.py).  
To use your own credentials, create a `.env` file in the project root:

```env
IBM_API_KEY=your_ibm_api_key_here
IBM_PROJECT_ID=your_project_id_here
IBM_ENDPOINT_URL=https://au-syd.ml.cloud.ibm.com
```

> Supported regional endpoints: `us-south`, `eu-de`, `au-syd`, `jp-tok`

### 5. Run the application

```bash
python app.py
```

The app opens at **http://localhost:5000**

---

## 📊 Data Import Guide

### CSV Format

Download a ready-to-use template from the **Data Import** page inside the app.

#### Required Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `date` | string | ISO datetime | `2024-05-01 09:00` |
| `platform` | string | Social platform | `Instagram`, `Facebook`, `LinkedIn`, `X (Twitter)` |
| `post_text` | string | Full post content | `Check out our new collection! #fashion` |
| `sentiment` | string | Sentiment label | `positive`, `negative`, `neutral` |
| `likes` | integer | Like count | `450` |
| `comments` | integer | Comment count | `32` |
| `shares` | integer | Share/retweet count | `18` |

#### Optional Columns

| Column | Description |
|--------|-------------|
| `hashtags` | Space-separated hashtags |
| `topic` | Content topic/category |
| `engagement` | Pre-calculated score (auto-computed if absent) |

> **Engagement formula**: `likes + (comments × 2) + (shares × 3)`  
> **Engagement rate** (%) requires follower counts — documented inline when used.

### Live API Integration

SocialSense AI does **not** pull live social media data by default. To integrate:

1. Obtain API credentials from your platform developer portal (Twitter/X, Instagram Graph API, LinkedIn API, etc.)
2. Extend `data/sample_data.py` with your API client
3. Return a DataFrame matching the schema above

---

## 🤖 IBM Granite Integration

| Setting | Value |
|---------|-------|
| Model | `ibm/granite-3-8b-instruct` |
| Endpoint | `https://au-syd.ml.cloud.ibm.com` |
| Max tokens | 700–1200 (varies by agent) |
| Temperature | 0.7 |

All AI calls include graceful fallback — if the model is unreachable, a clearly labelled placeholder is returned and the rest of the app continues to function normally.

---

## 🖼️ Multimodal Input

| Input Type | Status | Notes |
|------------|--------|-------|
| CSV files | ✅ Supported | Full validation, column mapping, engagement auto-calc |
| Images (PNG/JPG/WebP) | ✅ Upload supported | Visual analysis requires a multimodal Granite model |
| Audio (MP3/WAV/M4A) | ✅ Upload supported | Transcription requires IBM Watson Speech to Text |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Web Framework | [Flask 3](https://flask.palletsprojects.com) |
| Templating | [Jinja2](https://jinja.palletsprojects.com) |
| AI / LLM | [IBM Granite 3 8B Instruct](https://www.ibm.com/granite) via [watsonx.ai](https://www.ibm.com/watsonx) |
| Data Processing | [Pandas](https://pandas.pydata.org), [NumPy](https://numpy.org) |
| Visualisation | [Plotly](https://plotly.com/python/) (charts rendered client-side via CDN) |
| ML Utilities | [scikit-learn](https://scikit-learn.org) |
| Image Handling | [Pillow](https://pillow.readthedocs.io) |
| Environment | [python-dotenv](https://github.com/theskumar/python-dotenv) |

---

## 📋 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `IBM_API_KEY` | Yes | (demo key in config.py) | IBM Cloud API key |
| `IBM_PROJECT_ID` | Yes | (demo ID in config.py) | watsonx.ai project ID |
| `IBM_ENDPOINT_URL` | No | `https://au-syd.ml.cloud.ibm.com` | Regional endpoint |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m 'Add: your feature description'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [IBM Granite](https://www.ibm.com/granite) — Foundation model powering all AI features
- [IBM watsonx.ai](https://www.ibm.com/watsonx) — Enterprise AI platform
- [Flask](https://flask.palletsprojects.com) — Web framework
- [Plotly](https://plotly.com) — Interactive visualisations

---

<div align="center">
  <sub>Built with ❤️ using IBM Granite on watsonx.ai</sub>
</div>
