from pathlib import Path
import html
import re

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

CF_WEIGHT = 0.70
CONTENT_WEIGHT = 0.30
TOP_NEIGHBORS = 20

APP_NAME = "MovieLens"
AUTHOR = "Manish Chapagain"

st.set_page_config(
    page_title=f"{APP_NAME} • {AUTHOR}",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CINEMATIC DARK DESIGN SYSTEM
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
    --bg-void: #08080c;
    --bg-base: #0d0d14;
    --bg-elevated: #13131d;
    --bg-card: #171722;
    --bg-card-hover: #1e1e2d;
    --bg-input: #1a1a26;

    --text-primary: #f5f5f7;
    --text-secondary: #a1a1b5;
    --text-tertiary: #6b6b85;
    --text-disabled: #3d3d52;

    --border-subtle: rgba(255,255,255,0.06);
    --border-default: rgba(255,255,255,0.10);
    --border-strong: rgba(255,255,255,0.16);

    --accent: #ff5c5c;
    --accent-glow: rgba(255,92,92,0.35);
    --accent-soft: rgba(255,92,92,0.12);
    --accent-2: #ffb347;
    --accent-2-soft: rgba(255,179,71,0.14);

    --gold: #ffd93d;
    --gold-soft: rgba(255,217,61,0.12);

    --success: #4ade80;
    --success-soft: rgba(74,222,128,0.12);

    --violet: #a78bfa;
    --violet-soft: rgba(167,139,250,0.12);
}

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-secondary);
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

.stApp {
    background: var(--bg-void);
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(255,92,92,0.10), transparent),
        radial-gradient(ellipse 60% 40% at 100% 0%, rgba(167,139,250,0.06), transparent);
}

header[data-testid="stHeader"] {
    background: transparent !important;
    z-index: 10 !important;
}

.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 4rem !important;
    max-width: 1280px;
}

#MainMenu, footer { visibility: hidden; }

/* Custom scrollbar */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb {
    background: var(--bg-card-hover);
    border-radius: 99px;
    border: 2px solid var(--bg-base);
}
::-webkit-scrollbar-thumb:hover { background: var(--text-tertiary); }

/* ================= HERO ================= */
.hero-wrap {
    position: relative;
    border-radius: 24px;
    padding: 3rem 3rem 2.6rem;
    margin-bottom: 2rem;
    overflow: hidden;
    background: linear-gradient(135deg, #151520 0%, #0f0f1a 100%);
    border: 1px solid var(--border-subtle);
    isolation: isolate;
}

.hero-wrap::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -10%;
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, var(--accent-glow) 0%, transparent 60%);
    filter: blur(60px);
    z-index: -1;
    animation: pulseGlow 8s ease-in-out infinite;
}

.hero-wrap::after {
    content: '';
    position: absolute;
    bottom: -60%;
    right: -5%;
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, rgba(167,139,250,0.18) 0%, transparent 60%);
    filter: blur(70px);
    z-index: -1;
    animation: pulseGlow 10s ease-in-out infinite reverse;
}

@keyframes pulseGlow {
    0%, 100% { opacity: 0.6; transform: scale(1); }
    50% { opacity: 1; transform: scale(1.08); }
}

.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: var(--accent-soft);
    border: 1px solid rgba(255,92,92,0.25);
    color: var(--accent);
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    border-radius: 999px;
    padding: 0.4rem 0.95rem;
    margin-bottom: 1.4rem;
}

.hero-eyebrow .dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 8px var(--accent);
    animation: blink 2s ease-in-out infinite;
}

@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.35; }
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    color: var(--text-primary);
    font-size: 3.4rem;
    font-weight: 700;
    letter-spacing: -0.045em;
    line-height: 1.02;
    margin: 0 0 1rem 0;
}

.hero-title .gradient {
    background: linear-gradient(135deg, #ff5c5c 0%, #ffb347 50%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    color: var(--text-secondary);
    font-size: 1.08rem;
    line-height: 1.6;
    max-width: 620px;
    margin: 0 0 1.8rem 0;
    font-weight: 400;
}

.hero-meta {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    flex-wrap: wrap;
    padding-top: 1.6rem;
    border-top: 1px solid var(--border-subtle);
}

.hero-meta-item {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    font-size: 0.85rem;
    color: var(--text-secondary);
}

.hero-meta-item .icon {
    width: 32px; height: 32px;
    border-radius: 9px;
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.9rem;
}

.hero-meta-item strong {
    color: var(--text-primary);
    font-weight: 600;
}

/* ================= STATS ================= */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}

.stat-card {
    position: relative;
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 18px;
    padding: 1.5rem 1.6rem;
    overflow: hidden;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    opacity: 0;
    transition: opacity 0.3s;
}

.stat-card:hover {
    border-color: var(--border-strong);
    background: var(--bg-card-hover);
    transform: translateY(-3px);
    box-shadow: 0 20px 40px -20px rgba(0,0,0,0.8);
}

.stat-card:hover::before { opacity: 1; }

.stat-icon {
    width: 38px; height: 38px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.05rem;
    margin-bottom: 1.1rem;
}

.stat-icon.red { background: var(--accent-soft); border: 1px solid rgba(255,92,92,0.2); }
.stat-icon.gold { background: var(--gold-soft); border: 1px solid rgba(255,217,61,0.2); }
.stat-icon.violet { background: var(--violet-soft); border: 1px solid rgba(167,139,250,0.2); }

.stat-label {
    color: var(--text-tertiary);
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
}

.stat-value {
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-primary);
    font-size: 2.1rem;
    font-weight: 700;
    line-height: 1;
    letter-spacing: -0.02em;
    margin-bottom: 0.4rem;
}

.stat-sub {
    color: var(--text-tertiary);
    font-size: 0.8rem;
}


.model-panel {
    position: relative;
    background: linear-gradient(135deg, rgba(255,92,92,0.04) 0%, rgba(167,139,250,0.04) 100%);
    border: 1px solid var(--border-subtle);
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 2rem;
    overflow: hidden;
}

.model-panel::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: linear-gradient(180deg, var(--accent), var(--violet));
}

.model-header {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin-bottom: 0.9rem;
}

.model-header .badge {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--accent);
    background: var(--accent-soft);
    border: 1px solid rgba(255,92,92,0.25);
    border-radius: 6px;
    padding: 0.25rem 0.6rem;
}

.model-header h3 {
    color: var(--text-primary);
    font-size: 1rem;
    font-weight: 600;
    margin: 0;
}

.model-body {
    color: var(--text-secondary);
    font-size: 0.92rem;
    line-height: 1.7;
    margin: 0 0 1rem 0;
}

.model-body strong { color: var(--text-primary); font-weight: 600; }

.weight-bar {
    display: flex;
    height: 8px;
    border-radius: 99px;
    overflow: hidden;
    background: var(--bg-input);
    margin: 1rem 0 0.6rem;
}

.weight-bar .seg-cf {
    background: linear-gradient(90deg, #ff5c5c, #ff7b7b);
    height: 100%;
    transition: all 0.3s;
}

.weight-bar .seg-content {
    background: linear-gradient(90deg, #a78bfa, #c4b5fd);
    height: 100%;
    transition: all 0.3s;
}

.weight-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    color: var(--text-tertiary);
    font-family: 'JetBrains Mono', monospace;
}

.weight-labels span strong { color: var(--text-primary); }

.model-foot {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding-top: 1rem;
    margin-top: 1rem;
    border-top: 1px solid var(--border-subtle);
    color: var(--text-tertiary);
    font-size: 0.8rem;
}

/* ================= PRIMARY BUTTON ================= */
.st-key-main_recommend_btn button {
    position: relative !important;
    background: linear-gradient(135deg, #ff5c5c 0%, #ff7b4d 100%) !important;
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.95rem 2.6rem !important;
    box-shadow: 0 8px 32px -8px var(--accent-glow), inset 0 1px 0 rgba(255,255,255,0.15) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.st-key-main_recommend_btn button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 14px 44px -8px var(--accent-glow), inset 0 1px 0 rgba(255,255,255,0.2) !important;
}

.st-key-main_recommend_btn button:active {
    transform: translateY(0) !important;
}

/* ================= TABS ================= */
.stTabs [data-baseweb="tab-list"] {
    gap: 0 !important;
    background: var(--bg-card) !important;
    padding: 5px !important;
    border-radius: 14px !important;
    border: 1px solid var(--border-subtle) !important;
    border-bottom: 1px solid var(--border-subtle) !important;
    display: inline-flex !important;
    margin-bottom: 1.8rem !important;
}

.stTabs [data-baseweb="tab"] {
    height: 2.6rem !important;
    padding: 0.5rem 1.4rem !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.87rem !important;
    color: var(--text-tertiary) !important;
    background: transparent !important;
    border-radius: 10px !important;
    border: none !important;
    transition: all 0.2s ease !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-secondary) !important;
    background: rgba(255,255,255,0.03) !important;
}

.stTabs [aria-selected="true"] {
    background: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4), inset 0 0 0 1px var(--border-default) !important;
}

.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ================= MOVIE CARDS ================= */
.movie-card {
    position: relative;
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 18px;
    padding: 1.5rem 1.6rem;
    margin-bottom: 1rem;
    min-height: 200px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    overflow: hidden;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.movie-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    opacity: 0;
    transition: opacity 0.3s;
}

.movie-card:hover {
    border-color: var(--border-strong);
    background: var(--bg-card-hover);
    transform: translateY(-3px);
    box-shadow: 0 24px 48px -24px rgba(0,0,0,0.9), 0 0 0 1px rgba(255,92,92,0.05);
}

.movie-card:hover::before { opacity: 1; }

.card-top {
    display: flex;
    align-items: flex-start;
    gap: 0.85rem;
    margin-bottom: 0.9rem;
}

.rank-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--accent);
    background: var(--accent-soft);
    border: 1px solid rgba(255,92,92,0.22);
    border-radius: 7px;
    padding: 0.32rem 0.55rem;
    line-height: 1;
    flex-shrink: 0;
    letter-spacing: 0.02em;
}

.movie-title {
    color: var(--text-primary);
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.12rem;
    font-weight: 600;
    line-height: 1.35;
    margin: 0;
    flex: 1;
    letter-spacing: -0.015em;
}

.year-chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    color: var(--text-tertiary);
    background: var(--bg-input);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    padding: 0.22rem 0.5rem;
    flex-shrink: 0;
}

.genre-container {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    margin-bottom: 1.2rem;
}

.genre-pill {
    font-size: 0.71rem;
    font-weight: 500;
    color: var(--text-secondary);
    background: var(--bg-input);
    border: 1px solid var(--border-subtle);
    border-radius: 999px;
    padding: 0.22rem 0.65rem;
    letter-spacing: 0.01em;
}

.meter-track {
    width: 100%;
    height: 4px;
    background: var(--bg-input);
    border-radius: 99px;
    overflow: hidden;
    margin-bottom: 1rem;
}

.meter-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #ff5c5c 0%, #ffb347 100%);
    box-shadow: 0 0 12px var(--accent-glow);
    transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}

.card-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-top: 0.9rem;
    border-top: 1px solid var(--border-subtle);
}

.rating-badge {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.83rem;
    font-weight: 600;
    color: var(--gold);
    background: var(--gold-soft);
    border: 1px solid rgba(255,217,61,0.18);
    border-radius: 7px;
    padding: 0.28rem 0.65rem;
}

.rating-badge .star {
    font-size: 0.9rem;
    line-height: 1;
}

.rating-badge .pred-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.68rem;
    font-weight: 500;
    color: rgba(255,217,61,0.6);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.match-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--text-primary);
    background: linear-gradient(135deg, rgba(255,92,92,0.18), rgba(167,139,250,0.18));
    border: 1px solid rgba(255,92,92,0.2);
    border-radius: 7px;
    padding: 0.28rem 0.7rem;
}

/* ================= SIDEBAR ================= */
[data-testid="stSidebar"] {
    background: var(--bg-base) !important;
    border-right: 1px solid var(--border-subtle) !important;
}

[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

.side-brand {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding-bottom: 0.3rem;
}

.side-brand .logo {
    width: 34px; height: 34px;
    border-radius: 10px;
    background: linear-gradient(135deg, #ff5c5c, #ffb347);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    box-shadow: 0 4px 16px var(--accent-glow);
}

.side-brand .name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.02em;
}

.side-section {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-tertiary);
    margin: 1.1rem 0 0.5rem 0;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.side-section .count {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
    color: var(--text-tertiary);
    letter-spacing: 0;
}

.user-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 0.45rem 0.8rem;
    font-size: 0.79rem;
    font-weight: 500;
    color: var(--text-secondary);
    margin: 0.4rem 0 0.2rem 0;
}

.user-pill .dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--success);
    box-shadow: 0 0 8px var(--success);
}

.user-pill strong {
    color: var(--text-primary);
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}

.fav-item-row {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 9px;
    padding: 0.55rem 0.7rem;
    font-size: 0.79rem;
    font-weight: 500;
    color: var(--text-secondary);
    line-height: 1.35;
    word-break: break-word;
    transition: all 0.2s ease;
}

.fav-item-row:hover {
    border-color: var(--border-default);
    color: var(--text-primary);
}

/* Sidebar buttons (remove) */
div[data-testid="stSidebar"] div.stButton > button {
    border-radius: 7px !important;
    padding: 0.25rem 0.5rem !important;
    min-height: 2.05rem !important;
    height: 2.05rem !important;
    background: var(--bg-card) !important;
    color: var(--text-tertiary) !important;
    border: 1px solid var(--border-subtle) !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    box-shadow: none !important;
    transition: all 0.15s ease !important;
}

div[data-testid="stSidebar"] div.stButton > button:hover {
    background: var(--accent-soft) !important;
    color: var(--accent) !important;
    border-color: rgba(255,92,92,0.3) !important;
    transform: none !important;
}

/* Add to favorites button */
div[data-testid="stSidebar"] .st-key-add_movie_btn button {
    background: linear-gradient(135deg, #ff5c5c, #ff7b4d) !important;
    color: #ffffff !important;
    border: none !important;
    width: 100% !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    height: 2.4rem !important;
    border-radius: 9px !important;
    box-shadow: 0 4px 16px -4px var(--accent-glow) !important;
}
div[data-testid="stSidebar"] .st-key-add_movie_btn button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px -4px var(--accent-glow) !important;
}

/* Sidebar inputs */
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] > div,
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
    background: var(--bg-card) !important;
    border-color: var(--border-subtle) !important;
    border-radius: 9px !important;
    color: var(--text-secondary) !important;
    font-size: 0.84rem !important;
}

[data-testid="stSidebar"] [data-baseweb="select"]:hover > div {
    border-color: var(--border-default) !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] [role="listbox"] {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-default) !important;
}

[data-testid="stSidebar"] [data-baseweb="popover"] {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: 10px !important;
}

/* Slider */
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-soft) !important;
}

[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent-2)) !important;
}

[data-testid="stSidebar"] .stSlider [data-testid="stTickBar"] {
    color: var(--text-tertiary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
}

/* Sidebar divider */
[data-testid="stSidebar"] hr {
    border-color: var(--border-subtle) !important;
    margin: 0.8rem 0 !important;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.5rem !important; }
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] { gap: 0.4rem; align-items: center; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { margin-bottom: 0; }
[data-testid="stSidebarHeader"] { height: 2.2rem; min-height: 0; padding: 0.3rem 1rem 0 1rem; }

/* ================= MAIN AREA WIDGETS ================= */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
    overflow: hidden;
}
[data-testid="stExpander"] summary {
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}
[data-testid="stExpander"] summary:hover { color: var(--text-primary) !important; }

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    overflow: hidden;
}

/* Download button */
.stDownloadButton button {
    background: var(--bg-card) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.2s ease !important;
}
.stDownloadButton button:hover {
    background: var(--bg-card-hover) !important;
    color: var(--text-primary) !important;
    border-color: var(--border-default) !important;
}

/* Alerts / info */
[data-testid="stAlert"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
}

/* Section titles */
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: -0.025em;
    margin: 2.2rem 0 0.5rem 0;
    display: flex;
    align-items: center;
    gap: 0.7rem;
}

.section-title .accent-line {
    width: 3px;
    height: 24px;
    background: linear-gradient(180deg, var(--accent), var(--accent-2));
    border-radius: 2px;
}

.section-desc {
    color: var(--text-tertiary);
    font-size: 0.92rem;
    line-height: 1.6;
    margin-bottom: 1.5rem;
}

/* Footer */
.app-footer {
    margin-top: 4rem;
    padding: 2rem 0 0;
    border-top: 1px solid var(--border-subtle);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 1rem;
    color: var(--text-tertiary);
    font-size: 0.8rem;
}

.app-footer .brand {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    color: var(--text-secondary);
}

.app-footer .brand strong {
    color: var(--text-primary);
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
}

.app-footer .tech {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.02em;
}

</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data Loading (unchanged)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_data():
    ratings_path = DATA_DIR / "ratings.csv"
    movies_path = DATA_DIR / "movies.csv"
    tags_path = DATA_DIR / "tags.csv"

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    tags = pd.read_csv(tags_path)

    required_rating_cols = {"userId", "movieId", "rating"}
    required_movie_cols = {"movieId", "title", "genres"}
    required_tag_cols = {"movieId", "tag"}

    if not required_rating_cols.issubset(ratings.columns):
        raise ValueError("ratings.csv is missing required columns.")
    if not required_movie_cols.issubset(movies.columns):
        raise ValueError("movies.csv is missing required columns.")
    if not required_tag_cols.issubset(tags.columns):
        raise ValueError("tags.csv is missing required columns.")

    for column in ["userId", "movieId", "rating"]:
        ratings[column] = pd.to_numeric(ratings[column], errors="coerce")

    ratings = ratings.dropna(subset=["userId", "movieId", "rating"]).copy()
    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings = ratings[ratings["rating"].between(0.5, 5.0)].copy()

    movies = movies.drop_duplicates(subset=["movieId"]).copy()
    movies["movieId"] = pd.to_numeric(movies["movieId"], errors="coerce")
    movies = movies.dropna(subset=["movieId"]).copy()
    movies["movieId"] = movies["movieId"].astype(int)

    tags["movieId"] = pd.to_numeric(tags["movieId"], errors="coerce")
    tags = tags.dropna(subset=["movieId"]).copy()
    tags["movieId"] = tags["movieId"].astype(int)

    tags_by_movie = (
        tags.groupby("movieId")["tag"]
        .apply(lambda values: " ".join(values.dropna().astype(str)))
        .reset_index()
    )

    movies = movies.merge(tags_by_movie, on="movieId", how="left")
    movies["title"] = movies["title"].fillna("Unknown title")
    movies["genres"] = movies["genres"].fillna("(Genres unknown)")
    movies["tag"] = movies["tag"].fillna("")

    movies["content"] = (
        movies["genres"].str.replace("|", " ", regex=False) + " " + movies["tag"]
    ).str.strip()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        token_pattern=r"(?u)\b\w+\b",
        dtype=np.float32,
    )
    tfidf_matrix = vectorizer.fit_transform(movies["content"])

    user_movie = ratings.pivot_table(
        index="userId", columns="movieId", values="rating", aggfunc="mean",
    )

    rating_array = user_movie.to_numpy(dtype=np.float32)
    rating_mask = np.isfinite(rating_array)

    global_mean = float(ratings["rating"].mean())
    user_means = np.nanmean(rating_array, axis=1)
    user_means = np.nan_to_num(user_means, nan=global_mean)

    centered = np.where(rating_mask, rating_array - user_means[:, None], 0.0)

    user_similarity = cosine_similarity(centered).astype(np.float32)
    np.fill_diagonal(user_similarity, 0.0)

    movie_ids = movies["movieId"].to_numpy(dtype=int)
    user_ids = user_movie.index.to_numpy(dtype=int)
    rating_movie_ids = user_movie.columns.to_numpy(dtype=int)

    movie_to_index = {int(m): i for i, m in enumerate(movie_ids)}
    user_to_index = {int(u): i for i, u in enumerate(user_ids)}
    movie_to_rating_col = {int(m): i for i, m in enumerate(rating_movie_ids)}

    return {
        "ratings": ratings,
        "movies": movies,
        "user_movie": user_movie,
        "rating_array": rating_array,
        "rating_mask": rating_mask,
        "user_means": user_means,
        "user_similarity": user_similarity,
        "movie_ids": movie_ids,
        "user_ids": user_ids,
        "rating_movie_ids": rating_movie_ids,
        "movie_to_index": movie_to_index,
        "user_to_index": user_to_index,
        "movie_to_rating_col": movie_to_rating_col,
        "tfidf_matrix": tfidf_matrix,
        "global_mean": global_mean,
    }


def recommend(data, user_id, favorite_ids, limit=10):
    user_id = int(user_id)
    if user_id not in data["user_to_index"]:
        raise ValueError(f"Unknown user ID: {user_id}")

    user_index = data["user_to_index"][user_id]
    user_movie = data["user_movie"]
    user_ratings = user_movie.loc[user_id].dropna()

    seen_movies = {int(movie_id) for movie_id in user_ratings.index}

    valid_favorites = list(dict.fromkeys(
        int(movie_id)
        for movie_id in favorite_ids
        if int(movie_id) in data["movie_to_index"]
    ))

    if not valid_favorites:
        valid_favorites = [
            int(movie_id)
            for movie_id, rating in user_ratings.items()
            if rating >= 4.0 and int(movie_id) in data["movie_to_index"]
        ]

    if not valid_favorites:
        raise ValueError(
            "Choose at least one favourite movie, or select a user who rated movies 4 stars or higher."
        )

    liked_indices = [data["movie_to_index"][m] for m in valid_favorites]
    tfidf_matrix = data["tfidf_matrix"]
    user_profile = tfidf_matrix[liked_indices].mean(axis=0)

    content_scores = cosine_similarity(np.asarray(user_profile), tfidf_matrix).ravel()

    similarities = data["user_similarity"][user_index]
    neighbor_indices = np.argsort(similarities)[::-1][:TOP_NEIGHBORS]
    neighbor_indices = [
        idx for idx in neighbor_indices if idx != user_index and similarities[idx] > 0
    ]

    if not neighbor_indices:
        return [], []

    rating_array = data["rating_array"]
    rating_mask = data["rating_mask"]
    user_means = data["user_means"]
    rating_movie_ids = data["rating_movie_ids"]

    neighbor_weights = similarities[neighbor_indices]
    neighbor_ratings = rating_array[neighbor_indices]
    neighbor_masks = rating_mask[neighbor_indices]

    deviations = np.where(
        neighbor_masks,
        neighbor_ratings - user_means[neighbor_indices, None],
        0.0,
    )

    weighted_sum = (neighbor_weights[:, None] * deviations).sum(axis=0)
    available_weights = (neighbor_weights[:, None] * neighbor_masks).sum(axis=0)
    neighbor_counts = neighbor_masks.sum(axis=0)

    cf_predictions = np.full(len(rating_movie_ids), np.nan, dtype=np.float32)
    valid_columns = (available_weights > 0) & (neighbor_counts >= 2)

    cf_predictions[valid_columns] = (
        user_means[user_index]
        + weighted_sum[valid_columns] / available_weights[valid_columns]
    )

    candidates = []
    candidate_predictions = []
    candidate_content_scores = []

    for movie_id in data["movie_ids"]:
        movie_id = int(movie_id)
        if movie_id in seen_movies or movie_id in valid_favorites:
            continue

        rating_col = data["movie_to_rating_col"].get(movie_id)
        if rating_col is None:
            continue

        predicted = cf_predictions[rating_col]
        if not np.isfinite(predicted):
            continue

        candidates.append(movie_id)
        candidate_predictions.append(float(np.clip(predicted, 0.5, 5.0)))
        candidate_content_scores.append(float(content_scores[data["movie_to_index"][movie_id]]))

    if not candidates:
        return [], []

    candidate_content_scores = np.asarray(candidate_content_scores, dtype=np.float32)
    min_content = float(candidate_content_scores.min())
    max_content = float(candidate_content_scores.max())

    if max_content > min_content:
        normalized_content = (candidate_content_scores - min_content) / (max_content - min_content)
    else:
        normalized_content = np.zeros(len(candidates), dtype=np.float32)

    cf_results = []
    hybrid_results = []

    for movie_id, predicted_rating, content_score in zip(
        candidates, candidate_predictions, normalized_content,
    ):
        cf_normalized = (predicted_rating - 0.5) / 4.5
        hybrid_score = CF_WEIGHT * cf_normalized + CONTENT_WEIGHT * float(content_score)

        movie_row = data["movies"][data["movies"]["movieId"] == movie_id]
        if movie_row.empty:
            continue
        movie = movie_row.iloc[0]

        common = {
            "movieId": movie_id,
            "title": str(movie["title"]),
            "genres": str(movie["genres"]),
            "predicted_rating": round(predicted_rating, 2),
        }

        cf_results.append({**common, "score": round(predicted_rating, 4)})
        hybrid_results.append({**common, "score": round(float(hybrid_score), 4)})

    cf_results.sort(key=lambda item: item["predicted_rating"], reverse=True)
    hybrid_results.sort(key=lambda item: item["score"], reverse=True)

    return cf_results[:limit], hybrid_results[:limit]


# ---------------------------------------------------------------------------
# Card Display
# ---------------------------------------------------------------------------
def display_movies(results, download_key, is_hybrid=True):
    if not results:
        st.warning(
            "No qualifying movie suggestions found for this user. "
            "Try picking different favorite movies or another viewer."
        )
        return

    for start in range(0, len(results), 2):
        columns = st.columns(2, gap="medium")

        for offset, container in enumerate(columns):
            index = start + offset
            if index >= len(results):
                continue

            movie = results[index]
            raw_title = str(movie["title"])

            year_match = re.search(r"\((\d{4})\)$", raw_title.strip())
            if year_match:
                year_text = year_match.group(1)
                clean_title = raw_title[: year_match.start()].strip()
            else:
                year_text = ""
                clean_title = raw_title

            escaped_title = html.escape(clean_title)
            genres_list = [g.strip() for g in movie["genres"].split("|") if g.strip()]
            genre_badges = "".join(
                f'<span class="genre-pill">{html.escape(g)}</span>'
                for g in genres_list[:4]
            )

            if is_hybrid:
                match_pct = int(movie["score"] * 100)
                score_badge_html = f'<span class="match-badge">{match_pct}% Match</span>'
                bar_pct = match_pct
            else:
                bar_pct = int((movie["predicted_rating"] / 5.0) * 100)
                score_badge_html = f'<span class="match-badge">Rank #{index + 1}</span>'

            year_badge_html = f'<span class="year-chip">{year_text}</span>' if year_text else ""

            with container:
                st.markdown(
                    f"""
                    <div class="movie-card">
                        <div>
                            <div class="card-top">
                                <span class="rank-badge">#{index + 1:02d}</span>
                                <h3 class="movie-title">{escaped_title}</h3>
                                {year_badge_html}
                            </div>
                            <div class="genre-container">
                                {genre_badges}
                            </div>
                        </div>
                        <div>
                            <div class="meter-track">
                                <div class="meter-fill" style="width: {bar_pct}%;"></div>
                            </div>
                            <div class="card-footer">
                                <div class="rating-badge">
                                    <span class="star">★</span>
                                    <span>{movie['predicted_rating']:.1f} / 5</span>
                                    <span class="pred-label">predicted</span>
                                </div>
                                {score_badge_html}
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    table_data = pd.DataFrame(results)
    friendly_table = pd.DataFrame({
        "Movie ID": table_data["movieId"],
        "Title": table_data["title"],
        "Genres": table_data["genres"].str.replace("|", ", ", regex=False),
        "Predicted Rating (1-5)": table_data["predicted_rating"],
    })

    if is_hybrid:
        friendly_table["Hybrid Score (0-1)"] = table_data["score"]
        friendly_table["Match (%)"] = (table_data["score"] * 100.0).round(1)

    with st.expander("📊 View results as a table"):
        st.dataframe(friendly_table, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download as CSV",
        data=friendly_table.to_csv(index=False).encode("utf-8"),
        file_name=f"movielens_{download_key}_picks.csv",
        mime="text/csv",
        key=f"download_{download_key}",
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def add_favourite(label_to_id):
    label = st.session_state.get("add_choice")
    if label and label in label_to_id:
        movie_id = label_to_id[label]
        if movie_id not in st.session_state["fav_ids"]:
            st.session_state["fav_ids"].append(movie_id)
    st.session_state["add_choice"] = None


def remove_favourite(movie_id):
    st.session_state["fav_ids"] = [
        item for item in st.session_state["fav_ids"] if item != movie_id
    ]


st.markdown(
    f"""
    <div class="hero-wrap">
        <div class="hero-eyebrow">
            <span class="dot"></span>
            MovieLens - Hybrid Recommender
        </div>
        <h1 class="hero-title">
            Find your next <span class="gradient">obsession</span>.
        </h1>
        <p class="hero-sub">
            A dual-engine recommender blending collaborative community intelligence
            with deep content understanding — so every suggestion feels hand-picked.
        </p>
        <div class="hero-meta">
            <div class="hero-meta-item">
                <div class="icon">🎯</div>
                <div><strong>Hybrid Engine</strong> · CF + Content</div>
            </div>
            <div class="hero-meta-item">
                <div class="icon">⚡</div>
                <div><strong>Real-time</strong> · Instant scoring</div>
            </div>
            <div class="hero-meta-item">
                <div class="icon">🛡️</div>
                <div><strong>Neighborhood Guard</strong> · Min. 2 verified neighbors</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    with st.spinner("Warming up the engine…"):
        data = load_data()

    movies_df = data["movies"].drop_duplicates(subset=["movieId"]).copy()
    ratings_df = data["ratings"]

    # ---------- SIDEBAR ----------
    st.sidebar.markdown(
        '<div class="side-brand">'
        '<div class="logo">🎬</div>'
        f'<div class="name">{html.escape(APP_NAME)}</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.divider()

    user_ids = sorted(int(user_id) for user_id in data["user_ids"])

    st.sidebar.markdown('<div class="side-section">👤 Viewer Profile</div>', unsafe_allow_html=True)
    selected_user = st.sidebar.selectbox(
        "Select viewer",
        user_ids,
        label_visibility="collapsed",
    )

    user_ratings = ratings_df[ratings_df["userId"] == selected_user]
    st.sidebar.markdown(
        f'<div class="user-pill"><span class="dot"></span>'
        f'<strong>User #{selected_user}</strong> · {len(user_ratings):,} ratings</div>',
        unsafe_allow_html=True,
    )

    favourite_rated = user_ratings[user_ratings["rating"] >= 4.0].sort_values("rating", ascending=False)
    default_ids = [
        int(m) for m in favourite_rated["movieId"].tolist()[:5]
        if int(m) in data["movie_to_index"]
    ]

    if st.session_state.get("fav_user") != selected_user:
        st.session_state["fav_user"] = selected_user
        st.session_state["fav_ids"] = list(default_ids)
        st.session_state["add_choice"] = None

    movie_options = {
        f"{row.title} (ID: {int(row.movieId)})": int(row.movieId)
        for row in movies_df.itertuples(index=False)
    }
    id_to_label = {movie_id: label for label, movie_id in movie_options.items()}

    favourite_ids = [m for m in st.session_state["fav_ids"] if m in id_to_label]

    st.sidebar.markdown(
        f'<div class="side-section">⭐ Favorites <span class="count">({len(favourite_ids)})</span></div>',
        unsafe_allow_html=True,
    )

    if favourite_ids:
        try:
            holder = (
                st.sidebar.container(height=230) if len(favourite_ids) > 4
                else st.sidebar.container()
            )
        except TypeError:
            holder = st.sidebar.container()

        with holder:
            for movie_id in favourite_ids:
                col_txt, col_btn = st.columns([5, 1])
                col_txt.markdown(
                    f'<div class="fav-item-row">{html.escape(id_to_label[movie_id])}</div>',
                    unsafe_allow_html=True,
                )
                col_btn.button(
                    "✕",
                    key=f"rm_{movie_id}",
                    help="Remove movie",
                    on_click=remove_favourite,
                    args=(movie_id,),
                )
    else:
        st.sidebar.info("No favorites selected. Using viewer's top-rated movies.")

    st.sidebar.markdown('<div class="side-section">➕ Add Movie</div>', unsafe_allow_html=True)
    available_labels = [label for label, mid in movie_options.items() if mid not in favourite_ids]
    st.sidebar.selectbox(
        "Search movie to add",
        options=available_labels,
        index=None,
        placeholder="Type a movie name…",
        key="add_choice",
        label_visibility="collapsed",
    )
    st.sidebar.button(
        "＋ Add to Favorites",
        key="add_movie_btn",
        on_click=add_favourite,
        args=(movie_options,),
    )

    selected_favorites = list(favourite_ids)

    st.sidebar.divider()
    st.sidebar.markdown('<div class="side-section">🎚️ Result Count</div>', unsafe_allow_html=True)
    top_n = st.sidebar.slider(
        "Recommendations to show",
        min_value=5, max_value=20, value=10, step=5,
        label_visibility="collapsed",
    )

    st.sidebar.caption(f"Created by {html.escape(AUTHOR)}")

    # ---------- KPI STATS ----------
    st.markdown(
        f"""
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-icon red">🎞️</div>
                <div class="stat-label">Catalog</div>
                <div class="stat-value">{len(movies_df):,}</div>
                <div class="stat-sub">movies indexed</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon gold">⭐</div>
                <div class="stat-label">Ratings</div>
                <div class="stat-value">{len(ratings_df):,}</div>
                <div class="stat-sub">verified reviews</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon violet">👥</div>
                <div class="stat-label">Viewers</div>
                <div class="stat-value">{ratings_df['userId'].nunique():,}</div>
                <div class="stat-sub">profile vectors</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cf_pct = int(round(CF_WEIGHT * 100))
    content_pct = int(round(CONTENT_WEIGHT * 100))

    # ---------- MODEL PANEL ----------
    st.markdown(
        f"""
        <div class="model-panel">
            <div class="model-header">
                <span class="badge">Hybrid Engine</span>
                <h3>How recommendations are scored</h3>
            </div>
            <p class="model-body">
                Every movie gets a composite score blending <strong>collaborative filtering</strong>
                (what viewers with your taste rated highly) with <strong>content similarity</strong>
                (TF-IDF across genres & tags). Already-rated titles are automatically excluded.
            </p>
            <div class="weight-bar">
                <div class="seg-cf" style="width: {cf_pct}%;"></div>
                <div class="seg-content" style="width: {content_pct}%;"></div>
            </div>
            <div class="weight-labels">
                <span><strong>{cf_pct}%</strong> Collaborative Filtering</span>
                <span><strong>{content_pct}%</strong> Content Similarity</span>
            </div>
            <div class="model-foot">
                <span>🛡️</span>
                <span>Predictions require at least two positive-similarity neighbors who rated the title.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------- MAIN ACTION ----------
    if st.button("✨ Get Recommendations", key="main_recommend_btn"):
        try:
            with st.spinner("Calculating neighborhood similarities and hybrid vectors…"):
                cf_results, hybrid_results = recommend(
                    data, selected_user, selected_favorites, top_n,
                )

            st.session_state["cf_results"] = cf_results
            st.session_state["hybrid_results"] = hybrid_results
            st.session_state["results_user"] = selected_user
            st.session_state["results_favorites"] = sorted(selected_favorites)
            st.session_state["results_limit"] = top_n

        except Exception as error:
            st.error(f"Computation error: {error}")

    results_match = (
        st.session_state.get("results_user") == selected_user
        and st.session_state.get("results_favorites") == sorted(selected_favorites)
        and st.session_state.get("results_limit") == top_n
    )

    if results_match:
        st.markdown(
            '<div class="section-title">'
            '<span class="accent-line"></span>'
            'Your Recommendations'
            '</div>',
            unsafe_allow_html=True,
        )

        hybrid_tab, cf_tab = st.tabs([
            f"✨ Best Overall Mix · {len(st.session_state['hybrid_results'])}",
            f"👥 Similar Viewers Only · {len(st.session_state['cf_results'])}",
        ])

        with hybrid_tab:
            st.markdown(
                f'<p class="section-desc">'
                f'<strong>Hybrid model:</strong> {cf_pct}% collaborative + {content_pct}% content similarity. '
                f'Sorted by match strength.'
                f'</p>',
                unsafe_allow_html=True,
            )
            display_movies(
                st.session_state["hybrid_results"],
                download_key="hybrid",
                is_hybrid=True,
            )

        with cf_tab:
            st.markdown(
                '<p class="section-desc">'
                '<strong>Collaborative filtering:</strong> ranked strictly by predicted rating '
                'from viewers who share your taste.'
                '</p>',
                unsafe_allow_html=True,
            )
            display_movies(
                st.session_state["cf_results"],
                download_key="cf",
                is_hybrid=False,
            )
    else:
        st.info("👉 Click **'Get Recommendations'** above to generate your personalized picks.")

    # ---------- FOOTER ----------
    st.markdown(
        f"""
        <div class="app-footer">
            <div class="brand">
                <div class="stat-icon red" style="width:28px;height:28px;font-size:0.8rem;margin:0;">🎬</div>
                <span>Designed & built by <strong>{html.escape(AUTHOR)}</strong></span>
            </div>
            <div class="tech">Python · Streamlit · scikit-learn · Pandas · NumPy</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

except FileNotFoundError as error:
    st.error(
        "Dataset file not found. Ensure ratings.csv, movies.csv, "
        "and tags.csv are inside the 'data/' folder."
    )
    st.exception(error)

except Exception as error:
    st.error("The recommendation engine could not be loaded.")
    st.exception(error)