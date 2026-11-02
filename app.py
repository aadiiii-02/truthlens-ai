import os
import re
import html
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(
    page_title="TRUTHLENS AI | News Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --bg:#070b14; --panel:#0d1423; --line:rgba(148,163,184,.17); --cyan:#54e7ff; --violet:#9b8cff; --muted:#9aa9c1; }
html, body, [class*="css"] { font-family:'Manrope',sans-serif; }
.stApp { background: radial-gradient(ellipse at 70% 0%, rgba(42,66,140,.22), transparent 35%), #070b14; color:#eef4ff; }
[data-testid="stHeader"] { background:rgba(7,11,20,.72); }
.block-container { padding-top:1.5rem; padding-bottom:3rem; max-width:1450px; }
[data-testid="stSidebar"] { background:#0a101c; border-right:1px solid var(--line); }
[data-testid="stSidebar"] * { color:#eaf1ff; }
h1,h2,h3 { letter-spacing:-.035em; }
p,li { color:#c1cce0; }
.eyebrow { font-family:'DM Mono',monospace; color:var(--cyan); font-size:.76rem; letter-spacing:.16em; text-transform:uppercase; }
.hero { border:1px solid rgba(84,231,255,.22); border-radius:24px; padding:30px 32px; background:linear-gradient(120deg,rgba(17,31,55,.95),rgba(14,17,37,.82)); box-shadow:0 18px 70px rgba(0,0,0,.22); position:relative; overflow:hidden; }
.hero:after { content:''; position:absolute; width:280px;height:280px;right:-70px;top:-100px;border-radius:50%;border:1px solid rgba(84,231,255,.18);box-shadow:0 0 0 24px rgba(84,231,255,.025),0 0 0 50px rgba(155,140,255,.025); }
.hero-title { font-size:clamp(2rem,4vw,3.6rem); font-weight:800; line-height:1.02; color:#f4f8ff; margin:12px 0; }
.gradient { background:linear-gradient(90deg,#54e7ff,#a69aff);-webkit-background-clip:text;background-clip:text;color:transparent; }
.hero-copy { max-width:720px; font-size:1rem; line-height:1.7; color:#aebbd1; }
.pill { display:inline-block; border:1px solid rgba(84,231,255,.28); color:#8cefff; background:rgba(84,231,255,.07); padding:5px 10px; border-radius:100px; font-family:'DM Mono',monospace; font-size:.72rem; margin-right:6px; }
.metric-card { background:linear-gradient(145deg,rgba(17,27,46,.92),rgba(11,17,30,.92));border:1px solid var(--line);border-radius:16px;padding:18px 20px;min-height:110px; }
.metric-label { font-family:'DM Mono',monospace;color:#94a5c1;font-size:.72rem;text-transform:uppercase;letter-spacing:.08em; }
.metric-value { font-size:1.7rem;font-weight:800;color:#f3f7ff;margin-top:7px; }
.panel { border:1px solid var(--line);border-radius:18px;background:rgba(12,19,33,.78);padding:22px; }
.section-label { font-family:'DM Mono',monospace;color:#8fa2c0;font-size:.75rem;letter-spacing:.12em;text-transform:uppercase; }
div.stButton > button { border-radius:11px;border:1px solid rgba(84,231,255,.32);background:linear-gradient(90deg,#12617a,#4c3e91);color:white;font-weight:700;min-height:2.8rem; }
div.stButton > button:hover { border-color:#54e7ff;color:white;box-shadow:0 0 24px rgba(84,231,255,.12); }
.stTextArea textarea { background:#0a1120!important;color:#eaf1ff!important;border:1px solid rgba(148,163,184,.25)!important;border-radius:12px!important; }
div[data-baseweb="select"] > div { background:#0a1120;border-color:rgba(148,163,184,.25); }
hr { border-color:var(--line); }
.small-muted { color:#8d9db7;font-size:.82rem;line-height:1.6; }
.result-box { border:1px solid rgba(84,231,255,.24);background:linear-gradient(135deg,rgba(20,41,61,.7),rgba(22,20,46,.65));padding:22px;border-radius:18px; }
.warning-box { border:1px solid rgba(255,191,105,.28);background:rgba(255,191,105,.06);padding:13px 16px;border-radius:12px;color:#f2d9ad;font-size:.86rem; }
@keyframes pulseGlow { 0%,100% { box-shadow:0 0 12px rgba(84,231,255,.10), inset 0 0 10px rgba(84,231,255,.03); transform:translateY(0); } 50% { box-shadow:0 0 32px rgba(84,231,255,.26), inset 0 0 18px rgba(84,231,255,.06); transform:translateY(-2px); } }
@keyframes scanSweep { 0% { transform:translateY(-120%); opacity:0; } 15% { opacity:.9; } 85% { opacity:.9; } 100% { transform:translateY(260%); opacity:0; } }
@keyframes flickerIn { 0% { opacity:0; filter:blur(7px); transform:scale(.96); } 65% { opacity:1; filter:blur(0); transform:scale(1.015); } 100% { transform:scale(1); } }
@keyframes spinRadar { to { transform:rotate(360deg); } }
.scan-console { position:relative; overflow:hidden; border:1px solid rgba(84,231,255,.22); border-radius:18px; padding:18px; background:linear-gradient(145deg,rgba(9,27,43,.96),rgba(20,17,44,.95)); }
.scan-console:after { content:''; position:absolute; left:0; right:0; top:0; height:35%; background:linear-gradient(180deg,transparent,rgba(84,231,255,.12),transparent); animation:scanSweep 3.2s linear infinite; pointer-events:none; }
.radar { display:inline-flex; width:42px; height:42px; border:1px solid rgba(84,231,255,.6); border-radius:50%; align-items:center; justify-content:center; color:#54e7ff; box-shadow:0 0 20px rgba(84,231,255,.12); animation:spinRadar 7s linear infinite; }
.result-box { animation:flickerIn .65s ease-out both, pulseGlow 3s ease-in-out 0.7s infinite; }
.result-stamp { display:inline-block; padding:6px 12px; border-radius:8px; border:1px solid currentColor; letter-spacing:.14em; font:700 .72rem 'DM Mono',monospace; transform:rotate(-2deg); }
.metric-card { animation:flickerIn .5s ease-out both; }
div.stButton > button:active { transform:scale(.985); }


/* FUTURE INTERFACE LAYER */
.stApp {
  background:
    radial-gradient(ellipse at 12% 4%, rgba(0,220,255,.10), transparent 27%),
    radial-gradient(ellipse at 88% 12%, rgba(129,91,255,.13), transparent 29%),
    linear-gradient(rgba(84,231,255,.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(84,231,255,.025) 1px, transparent 1px),
    #050811 !important;
  background-size:auto,auto,42px 42px,42px 42px,auto !important;
}
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #080f1d 0%, #090b17 65%, #0b1020 100%) !important;
  box-shadow: 10px 0 55px rgba(0,0,0,.22);
}
.hero {
  background:
    radial-gradient(circle at 88% 28%, rgba(84,231,255,.12), transparent 20%),
    linear-gradient(115deg, rgba(12,26,48,.97), rgba(12,13,31,.96)) !important;
  box-shadow: 0 0 0 1px rgba(84,231,255,.035), 0 24px 90px rgba(0,0,0,.34), inset 0 1px rgba(255,255,255,.035) !important;
  animation: heroEnter .8s cubic-bezier(.2,.8,.2,1) both;
}
.hero:before {
  content:""; position:absolute; width:340px; height:340px; right:-90px; top:-115px;
  border:1px solid rgba(84,231,255,.22); border-radius:50%;
  box-shadow:0 0 0 18px rgba(84,231,255,.025),0 0 0 42px rgba(155,140,255,.025),0 0 65px rgba(84,231,255,.08);
  animation: orbitFloat 9s ease-in-out infinite;
}
.hero:after {
  content:""; position:absolute; width:210px;height:210px;right:-24px;top:-52px;border-radius:50%;
  border:1px dashed rgba(155,140,255,.28); animation: spinRadar 24s linear infinite;
}
.hero-title { text-shadow:0 0 34px rgba(84,231,255,.08); }
.metric-card {
  position:relative; overflow:hidden;
  background:linear-gradient(145deg,rgba(16,28,49,.94),rgba(8,14,27,.97)) !important;
  border:1px solid rgba(148,163,184,.16) !important;
  box-shadow:inset 0 1px rgba(255,255,255,.035), 0 12px 35px rgba(0,0,0,.14);
  transition:transform .22s ease,border-color .22s ease,box-shadow .22s ease;
}
.metric-card:after {
  content:""; position:absolute; left:-30%; top:0; width:35%; height:2px;
  background:linear-gradient(90deg,transparent,#54e7ff,transparent);
  animation: cardScan 5s linear infinite;
}
.metric-card:hover { transform:translateY(-4px); border-color:rgba(84,231,255,.4)!important; box-shadow:0 14px 35px rgba(0,0,0,.25),0 0 24px rgba(84,231,255,.07); }
.panel {
  background:linear-gradient(145deg,rgba(12,21,38,.91),rgba(8,13,25,.93)) !important;
  box-shadow:inset 0 1px rgba(255,255,255,.025),0 15px 38px rgba(0,0,0,.13);
}
.scan-console {
  background:
    radial-gradient(circle at 85% 20%,rgba(84,231,255,.12),transparent 24%),
    linear-gradient(145deg,rgba(7,25,40,.98),rgba(17,12,39,.98)) !important;
  box-shadow:inset 0 0 40px rgba(84,231,255,.025),0 0 28px rgba(84,231,255,.035);
}
.radar { position:relative; }
.radar:after { content:"";position:absolute;inset:-7px;border:1px dashed rgba(84,231,255,.3);border-radius:50%;animation:spinRadar 12s linear infinite; }
.result-box {
  border-color:rgba(84,231,255,.36) !important;
  box-shadow:inset 0 0 45px rgba(84,231,255,.035),0 0 38px rgba(84,231,255,.055);
}
div.stButton > button {
  background:linear-gradient(100deg,#087e98,#344cc2 58%,#7148c9) !important;
  border:1px solid rgba(104,233,255,.48) !important;
  box-shadow:0 5px 22px rgba(44,118,210,.16),inset 0 1px rgba(255,255,255,.14);
  transition:all .2s ease !important;
}
div.stButton > button:hover {
  transform:translateY(-2px);
  box-shadow:0 8px 30px rgba(84,231,255,.2),0 0 0 3px rgba(84,231,255,.045) !important;
}
.stTextArea textarea:focus { border-color:rgba(84,231,255,.65)!important;box-shadow:0 0 0 3px rgba(84,231,255,.07)!important; }
@keyframes heroEnter { from { opacity:0; transform:translateY(12px); filter:blur(5px); } to { opacity:1; transform:translateY(0); filter:blur(0); } }
@keyframes orbitFloat { 0%,100% { transform:translateY(0) scale(1); } 50% { transform:translateY(12px) scale(1.025); } }
@keyframes cardScan { 0% { left:-35%; opacity:0; } 15% { opacity:.9; } 55% { opacity:.5; } 100% { left:120%; opacity:0; } }
@media (prefers-reduced-motion: reduce) { *,*:before,*:after { animation-duration:.01ms!important; animation-iteration-count:1!important; transition-duration:.01ms!important; } }


/* RESULT ALERT: restrained, high-contrast and accessible */
.fake-alert {
  position:relative; display:flex; gap:15px; align-items:flex-start;
  margin:0 0 16px; padding:17px 19px; overflow:hidden;
  border:1px solid rgba(255,94,120,.65); border-left:4px solid #ff5d78;
  border-radius:15px; color:#ffeaf0;
  background:linear-gradient(110deg,rgba(72,18,39,.82),rgba(23,15,31,.94));
  box-shadow:0 0 0 1px rgba(255,93,120,.08),0 0 26px rgba(255,65,105,.12);
  animation: alertEnter .48s ease-out both, alertBreath 2.8s ease-in-out .5s infinite;
}
.fake-alert:after { content:""; position:absolute; top:0; bottom:0; left:-45%; width:32%;
  background:linear-gradient(90deg,transparent,rgba(255,128,151,.10),transparent);
  transform:skewX(-18deg); animation:alertSweep 4.5s ease-in-out 1s infinite; pointer-events:none; }
.alert-icon { flex:0 0 37px; width:37px;height:37px;display:grid;place-items:center;
  border-radius:11px;color:#ff7087;background:rgba(255,93,120,.12);
  border:1px solid rgba(255,93,120,.36);font-size:21px;animation:iconPulse 1.8s ease-in-out infinite; }
.alert-kicker { font:500 .68rem 'DM Mono',monospace;letter-spacing:.14em;text-transform:uppercase;color:#ff8da0;margin-bottom:4px; }
.alert-title { font-size:1.13rem;font-weight:800;letter-spacing:-.02em;color:#fff0f3;margin-bottom:4px; }
.alert-copy { color:#e6bac5;font-size:.88rem;line-height:1.55; }
.real-alert { display:flex;gap:12px;align-items:flex-start;margin:0 0 16px;padding:15px 17px;
  border:1px solid rgba(84,231,255,.30);border-left:3px solid #54e7ff;border-radius:14px;
  background:linear-gradient(110deg,rgba(10,49,59,.65),rgba(12,19,33,.92));animation:alertEnter .45s ease-out both; }
.real-alert .alert-icon { color:#54e7ff;background:rgba(84,231,255,.08);border-color:rgba(84,231,255,.25);animation:none; }
@keyframes alertEnter { from {opacity:0;transform:translateY(8px);filter:blur(3px)} to {opacity:1;transform:translateY(0);filter:blur(0)} }
@keyframes alertBreath { 0%,100% {box-shadow:0 0 0 1px rgba(255,93,120,.06),0 0 15px rgba(255,65,105,.08)} 50% {box-shadow:0 0 0 1px rgba(255,93,120,.16),0 0 25px rgba(255,65,105,.18)} }
@keyframes iconPulse { 0%,100% {transform:scale(1)} 50% {transform:scale(1.06)} }
@keyframes alertSweep { 0%,35% {left:-45%;opacity:0} 48% {opacity:1} 72%,100% {left:130%;opacity:0} }
@media (prefers-reduced-motion: reduce) { .fake-alert,.real-alert,.alert-icon,.fake-alert:after {animation:none!important} }

footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)
st.markdown("""<style>
/* TRUTHLENS reference-inspired premium black interface: A + B + C + D */
:root { --bg:#050506; --panel:#0b0b0e; --line:rgba(255,255,255,.12); --cyan:#ff4d5f; --violet:#ff4d5f; --muted:#a4a4ad; }
.stApp { background: radial-gradient(ellipse at 82% 5%,rgba(255,40,60,.10),transparent 28%),radial-gradient(ellipse at 10% 95%,rgba(255,40,60,.06),transparent 30%),#050506!important; color:#f6f6f7; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#080809,#0b080a 70%,#100809)!important; border-right:1px solid rgba(255,65,80,.2)!important; }
.eyebrow,.section-label { color:#ff6572!important; }
.hero { border:1px solid rgba(255,65,80,.24)!important; background:linear-gradient(120deg,rgba(18,15,17,.98),rgba(8,8,10,.96))!important; box-shadow:0 18px 70px rgba(0,0,0,.34)!important; }
.hero:after { border-color:rgba(255,65,80,.22)!important; box-shadow:0 0 0 24px rgba(255,65,80,.025),0 0 0 50px rgba(255,65,80,.02)!important; }
.gradient { background:linear-gradient(90deg,#fff,#ff5667)!important;-webkit-background-clip:text!important;background-clip:text!important;color:transparent!important; }
.pill { border-color:rgba(255,80,95,.3)!important;color:#ff9aa3!important;background:rgba(255,65,80,.06)!important; }
.metric-card,.panel { background:linear-gradient(145deg,rgba(18,17,19,.96),rgba(9,9,11,.96))!important;border-color:rgba(255,255,255,.11)!important; }
div.stButton > button { background:linear-gradient(100deg,#3b1117,#171215)!important;border:1px solid rgba(255,76,91,.5)!important;color:#fff!important; }
div.stButton > button:hover { border-color:#ff5969!important;box-shadow:0 0 25px rgba(255,55,75,.17)!important; }
.stTextArea textarea,div[data-baseweb="select"]>div { background:#0b0b0e!important;border-color:rgba(255,255,255,.16)!important;color:#f4f4f5!important; }
.scan-console { background:linear-gradient(145deg,#110d10,#09090b)!important;border-color:rgba(255,70,85,.24)!important; }
.scan-console:after { background:linear-gradient(180deg,transparent,rgba(255,65,80,.12),transparent)!important; }
.radar { border-color:rgba(255,90,105,.7)!important;color:#ff6b79!important;box-shadow:0 0 20px rgba(255,70,85,.16)!important; }
.result-box { border-color:rgba(255,70,85,.25)!important;background:linear-gradient(135deg,rgba(32,15,19,.7),rgba(12,11,13,.85))!important; }
.warning-box { border-color:rgba(255,174,74,.32)!important;background:rgba(255,174,74,.06)!important; }
.fake-alert { animation:flickerIn .55s ease-out both,pulseGlowRed 2.4s ease-in-out .5s infinite!important; }
@keyframes pulseGlowRed { 0%,100%{box-shadow:0 0 12px rgba(255,55,75,.08),inset 0 0 10px rgba(255,55,75,.02)}50%{box-shadow:0 0 32px rgba(255,55,75,.23),inset 0 0 18px rgba(255,55,75,.05)} }
.real-alert { border-color:rgba(62,220,130,.35)!important;background:linear-gradient(120deg,rgba(9,35,23,.75),rgba(10,13,12,.9))!important; }
.live-card {border:1px solid rgba(255,255,255,.12);border-radius:15px;background:linear-gradient(135deg,#121114,#09090b);padding:17px 18px;margin:10px 0;}
.live-card a {color:#ff8791;text-decoration:none}.live-meta{font: .72rem 'DM Mono',monospace;color:#9c9ca5;letter-spacing:.05em;text-transform:uppercase}
</style>""", unsafe_allow_html=True)

# ---------- Small built-in demonstration dataset ----------
# This is deliberately labelled as a demonstration set, not a validated news corpus.
DEMO_ROWS = [
    ("City council approves new public transport budget after a three-hour meeting", "REAL"),
    ("University announces revised examination schedule on its official website", "REAL"),
    ("Weather department issues heavy rainfall advisory for several districts", "REAL"),
    ("Local hospital opens a new emergency care wing, officials say", "REAL"),
    ("Central bank publishes its quarterly monetary policy statement", "REAL"),
    ("Researchers publish a peer-reviewed study on urban air quality", "REAL"),
    ("Railway services resume after maintenance work is completed", "REAL"),
    ("Election commission releases polling dates for the upcoming election", "REAL"),
    ("Government releases annual economic report with sector-wise figures", "REAL"),
    ("School board announces results through its official portal", "REAL"),
    ("BREAKING!!! Secret miracle drink cures every disease overnight", "FAKE"),
    ("You will not believe this shocking secret that officials are hiding", "FAKE"),
    ("Famous actor secretly becomes president according to anonymous post", "FAKE"),
    ("Scientists confirm humans can live forever after eating one fruit", "FAKE"),
    ("Share immediately before the government deletes this unbelievable truth", "FAKE"),
    ("A hidden planet will destroy Earth tomorrow, viral post claims", "FAKE"),
    ("Doctors hate this one strange trick that fixes every medical problem", "FAKE"),
    ("Anonymous source reveals impossible discovery with no evidence provided", "FAKE"),
    ("Miracle pill guarantees instant weight loss for everyone, post claims", "FAKE"),
    ("Secret group controls every world event, viral message alleges", "FAKE"),
    ("Officials say a new bridge project is expected to open next year", "REAL"),
    ("Public health agency updates its seasonal vaccination guidance", "REAL"),
    ("Report finds household energy costs rose during the past quarter", "REAL"),
    ("Court publishes its decision in a widely followed public case", "REAL"),
    ("Scientists test a new battery design in a laboratory study", "REAL"),
    ("URGENT!!! Click now to learn the forbidden truth they fear", "FAKE"),
    ("This unbelievable secret will change your life in just one day", "FAKE"),
    ("Viral message claims cash is being sent to everyone who forwards it", "FAKE"),
    ("No proof needed: anonymous post claims a celebrity faked everything", "FAKE"),
    ("Shocking claim spreads online without a named source or supporting evidence", "FAKE"),
]

@st.cache_data
def get_demo_df():
    return pd.DataFrame(DEMO_ROWS, columns=["text", "label"])

@st.cache_data(show_spinner=False, ttl=86400)
def load_public_isot_sample(rows_per_class=2500):
    """
    Fetch a capped sample from the public ISOT Fake/Real News CSV mirrors.
    This is a research/demo corpus, not a live fact-checking feed.
    """
    urls = {
        "REAL": "https://raw.githubusercontent.com/Reinforz/Fake-News-detection-with-ISOT-Dataset/main/True.csv",
        "FAKE": "https://raw.githubusercontent.com/Reinforz/Fake-News-detection-with-ISOT-Dataset/main/Fake.csv",
    }
    parts = []
    for label, url in urls.items():
        raw = pd.read_csv(url, nrows=rows_per_class, on_bad_lines="skip")
        title_col = next((c for c in raw.columns if c.lower().strip() == "title"), None)
        text_col = next((c for c in raw.columns if c.lower().strip() == "text"), None)
        if text_col is None:
            raise ValueError("The public dataset source did not contain an expected text column.")
        if title_col:
            combined = raw[title_col].fillna("").astype(str).str.strip() + ". " + raw[text_col].fillna("").astype(str).str.strip()
        else:
            combined = raw[text_col].fillna("").astype(str)
        part = pd.DataFrame({"text": combined, "label": label})
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    result["text"] = result["text"].str.replace(r"\\s+", " ", regex=True).str.strip()
    result = result[(result["text"].str.len() > 40)].drop_duplicates(subset=["text"])
    return result.reset_index(drop=True)

def normalize_label(v):
    s = str(v).strip().lower()
    if s in {"fake", "false", "1", "unreliable", "misleading"}:
        return "FAKE"
    if s in {"real", "true", "0", "reliable", "genuine"}:
        return "REAL"
    return None

def load_dataset(uploaded_file):
    if uploaded_file is None:
        local_path = "news.csv"
        if os.path.exists(local_path):
            df = pd.read_csv(local_path)
            source = "Local news.csv"
        else:
            return get_demo_df(), "Built-in demonstration examples", True
    else:
        df = pd.read_csv(uploaded_file)
        source = uploaded_file.name

    cols = {c.lower().strip(): c for c in df.columns}
    text_col = next((cols[c] for c in ["text", "article", "content", "title", "news", "headline"] if c in cols), None)
    label_col = next((cols[c] for c in ["label", "labels", "class", "target", "fake", "truth"] if c in cols), None)
    if text_col is None or label_col is None:
        raise ValueError("CSV needs a text/article/content/title column and a label/class column.")
    clean = pd.DataFrame({"text": df[text_col].fillna("").astype(str), "label": df[label_col].map(normalize_label)})
    clean = clean[(clean["text"].str.len() > 15) & clean["label"].notna()].drop_duplicates()
    if len(clean) < 10 or clean["label"].nunique() < 2:
        raise ValueError("Need at least 10 usable rows and examples from both REAL and FAKE classes.")
    return clean.reset_index(drop=True), source, False

@st.cache_resource
def train_model(texts_tuple, labels_tuple):
    texts = list(texts_tuple)
    labels = list(labels_tuple)
    counts = pd.Series(labels).value_counts()
    can_split = len(texts) >= 20 and counts.min() >= 2
    if can_split:
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.25, random_state=42, stratify=labels
        )
    else:
        X_train, X_test, y_train, y_test = texts, texts, labels, labels
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=25000, min_df=1)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, pos_label="FAKE", zero_division=0),
        "recall": recall_score(y_test, pred, pos_label="FAKE", zero_division=0),
        "f1": f1_score(y_test, pred, pos_label="FAKE", zero_division=0),
        "matrix": confusion_matrix(y_test, pred, labels=["REAL", "FAKE"]),
        "test_size": len(y_test),
        "split": can_split,
    }
    return model, metrics

def explain_words(model, text, predicted_label, limit=8):
    vectorizer = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]
    vec = vectorizer.transform([text])
    names = vectorizer.get_feature_names_out()
    # Logistic regression binary decision scores: positive weights point toward classes_[1].
    if len(clf.classes_) != 2:
        return []
    class_one = clf.classes_[1]
    direction = 1 if predicted_label == class_one else -1
    contributions = vec.toarray()[0] * clf.coef_[0] * direction
    idx = contributions.argsort()[::-1]
    found = []
    for i in idx:
        if contributions[i] <= 0:
            break
        token = names[i]
        if token not in text.lower():
            continue
        found.append((token, float(contributions[i])))
        if len(found) >= limit:
            break
    return found

# ---------- Recent news RSS helpers ----------
def fetch_google_news(query="India", limit=10):
    """Fetch recent headlines via Google News RSS; availability depends on network access."""
    q = urllib.parse.quote(query.strip() or "India")
    url = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 TRUTHLENS-AI/1.0"})
    with urllib.request.urlopen(req, timeout=12) as response:
        root = ET.fromstring(response.read())
    items=[]
    for item in root.findall("./channel/item")[:limit]:
        items.append({"title": item.findtext("title", "Untitled"), "link": item.findtext("link", ""), "published": item.findtext("pubDate", "Date unavailable"), "source": (item.findtext("source", "") or "News source")})
    return items

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown('<div class="eyebrow">TRUTHLENS / SYSTEM</div>', unsafe_allow_html=True)
    st.markdown("## 🛰️ Mission Control")
    if "navigation_target" in st.session_state:
        st.session_state["nav_page"] = st.session_state.pop("navigation_target")
    page = st.radio("NAVIGATION", ["Home", "Scanner", "Live News", "Model Lab", "About"], index=0, key="nav_page", label_visibility="visible")
    st.divider()
    st.markdown('<div class="section-label">SYSTEM STATUS</div>', unsafe_allow_html=True)
    st.markdown("🟢 Interface online")
    st.markdown("🔵 Local ML pipeline")
    st.markdown("🟡 Live news: RSS search available")
    st.divider()
    st.markdown('<div class="small-muted">Build 3.0 · BLACK GLASS INTERFACE<br>Designed for responsible media literacy.</div>', unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
<div class="hero">
  <div class="eyebrow">◉ TRUTHLENS NEURAL SCANNER &nbsp; / &nbsp; BLACK GLASS BUILD 3.0</div>
  <div class="hero-title">Decode the <span class="gradient">signal.</span></div>
  <div class="hero-copy">An AI-assisted news intelligence workspace. Explore recent headlines, screen claims, and inspect evidence sources before sharing.</div>
  <div style="margin-top:20px"><span class="pill">NLP ENGINE</span><span class="pill">TF-IDF + LOGISTIC REGRESSION</span><span class="pill">EXPLAINABLE SIGNALS</span></div>
</div>
""", unsafe_allow_html=True)
st.write("")

# Dataset upload is available from Scanner and Model Lab.
with st.sidebar:
    dataset_mode = st.radio(
        "DATA SOURCE",
        ["Public ISOT sample (~5,000 articles)", "Built-in mini demo", "Upload my own CSV"],
        index=0,
        help="The public option fetches up to 2,500 real-labelled and 2,500 fake-labelled older English news articles when the app starts."
    )
    uploaded = None
    if dataset_mode == "Upload my own CSV":
        uploaded = st.file_uploader("Upload a labelled CSV dataset", type=["csv"], help="Accepted columns include text/article/content/title and label/class. Labels should identify REAL and FAKE examples.")
    st.caption("Public benchmark data is historical and mainly English-language. Predictions are pattern-based, not live fact-checks.")

try:
    if dataset_mode == "Public ISOT sample (~5,000 articles)":
        df = load_public_isot_sample(2500)
        dataset_source = "ISOT public research dataset sample"
        is_demo = False
    elif dataset_mode == "Built-in mini demo":
        df, dataset_source, is_demo = get_demo_df(), "Built-in demonstration examples", True
    else:
        df, dataset_source, is_demo = load_dataset(uploaded)
    model, metrics = train_model(tuple(df["text"].tolist()), tuple(df["label"].tolist()))
    model_error = None
except Exception as e:
    df, dataset_source, is_demo, model, metrics = None, "Unavailable", False, None, None
    model_error = str(e)

if model_error:
    st.error(f"Dataset error: {model_error}")
    st.stop()

# ---------- Top stats ----------
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Training examples</div><div class="metric-value">{len(df):,}</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Model accuracy*</div><div class="metric-value">{metrics["accuracy"]*100:.1f}%</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Fake-class F1*</div><div class="metric-value">{metrics["f1"]:.2f}</div></div>', unsafe_allow_html=True)
with m4:
    status = "DEMO DATA" if is_demo else "CUSTOM DATA"
    st.markdown(f'<div class="metric-card"><div class="metric-label">Dataset mode</div><div class="metric-value" style="font-size:1.25rem">{status}</div></div>', unsafe_allow_html=True)
st.write("")
if is_demo:
    st.markdown('<div class="warning-box">⚠ MINI DEMO MODE — these examples only demonstrate the interface. The metrics do not measure real-world fake-news detection performance.</div>', unsafe_allow_html=True)
elif dataset_source.startswith("ISOT"):
    st.markdown('<div class="warning-box">RESEARCH DATASET MODE — using a capped sample from the historical ISOT corpus (mostly 2016–2017 English news). It is larger than the mini demo, but it is not current news and can contain source/topic bias. Metrics describe this sample only.</div>', unsafe_allow_html=True)
elif not metrics["split"]:
    st.markdown('<div class="warning-box">The dataset is small, so metrics are calculated on training examples and are not an independent test. Use more labelled examples for a reliable evaluation.</div>', unsafe_allow_html=True)

if page == "Home":
    st.markdown('<div class="section-label">INTELLIGENCE OVERVIEW / 00</div>', unsafe_allow_html=True)
    st.markdown("## Verify the signal. Not just the headline.")
    st.write("Choose a workflow to begin. TRUTHLENS combines an experimental text classifier with recent-news discovery; evidence must still be checked carefully.")
    c1,c2=st.columns(2,gap="large")
    with c1:
        st.markdown('<div class="live-card"><div class="live-meta">01 / CLAIM SCANNER</div><h3>Headline Verification</h3><p>Screen a short claim or headline with the text model, then investigate relevant reports.</p></div>',unsafe_allow_html=True)
        if st.button("OPEN SCANNER →",use_container_width=True):
            st.session_state["navigation_target"]="Scanner"
            st.rerun()
    with c2:
        st.markdown('<div class="live-card"><div class="live-meta">02 / ARTICLE INTELLIGENCE</div><h3>Full Article Analysis</h3><p>Paste an article to inspect learned text patterns and use its key claim as a search query.</p></div>',unsafe_allow_html=True)
        if st.button("ANALYSE ARTICLE →",use_container_width=True):
            st.session_state["navigation_target"]="Scanner"
            st.rerun()
    st.markdown("### Live news discovery")
    query=st.text_input("Search recent headlines",value="India",key="home_news_query",placeholder="e.g. India space mission, climate, education")
    if st.button("FETCH RECENT NEWS",use_container_width=True,key="home_fetch_news"):
        try:
            items=fetch_google_news(query,6)
            st.session_state["recent_news_items"]=items
            st.session_state["recent_news_query"]=query
        except Exception as e:
            st.warning("Could not reach the news feed right now. Please retry later. Network access may be unavailable on the hosting service.")
    for item in st.session_state.get("recent_news_items",[]):
        st.markdown(f'<div class="live-card"><div class="live-meta">{html.escape(item["source"])} · {html.escape(item["published"])}</div><h4>{html.escape(item["title"])}</h4><a href="{html.escape(item["link"],quote=True)}" target="_blank">OPEN SOURCE ↗</a></div>',unsafe_allow_html=True)

elif page == "Live News":
    st.markdown('<div class="section-label">LIVE NEWS / DISCOVERY FEED</div>',unsafe_allow_html=True)
    st.header("Recent headlines, ready to investigate")
    st.caption("Headlines are fetched from Google News RSS. A headline appearing here does not mean it has been independently verified.")
    query=st.text_input("Search topic or claim",value=st.session_state.get("recent_news_query","India"),key="live_news_query",placeholder="Search a headline, person, event, or claim")
    col_a,col_b=st.columns([1,3])
    with col_a:
        count=st.selectbox("Headlines",[5,10,15,20],index=1)
    with col_b:
        if st.button("↻ SEARCH LIVE NEWS",use_container_width=True):
            try:
                st.session_state["recent_news_items"]=fetch_google_news(query,count)
                st.session_state["recent_news_query"]=query
                st.success(f"Loaded headlines for: {query}")
            except Exception:
                st.error("News feed could not be reached. Try again later; this can happen if the host blocks external RSS requests.")
    for item in st.session_state.get("recent_news_items",[]):
        st.markdown(f'<div class="live-card"><div class="live-meta">{html.escape(item["source"])} · {html.escape(item["published"])}</div><h4>{html.escape(item["title"])}</h4><a href="{html.escape(item["link"],quote=True)}" target="_blank">READ REPORT ↗</a></div>',unsafe_allow_html=True)
    st.markdown('<div class="warning-box">EVIDENCE NOTE: This feed discovers reports; it does not prove that a claim is true. The current ML classifier is a text-pattern screening tool, not an independent fact-checking engine.</div>',unsafe_allow_html=True)

elif page == "Scanner":
    left, right = st.columns([1.2, .8], gap="large")
    with left:
        st.markdown('<div class="section-label">01 / INPUT CHANNEL</div>', unsafe_allow_html=True)
        st.subheader("Run a news scan")
        sample = st.selectbox("Quick test sample", ["Choose a sample…", "Official-style local news", "Sensational viral claim"])
        default_text = ""
        if sample == "Official-style local news":
            default_text = "The weather department issued a rainfall advisory for several districts on Tuesday."
        elif sample == "Sensational viral claim":
            default_text = "SHOCKING!!! Share this secret miracle cure before officials delete the truth!"
        article = st.text_area("Paste a headline or article", value=default_text, height=230, placeholder="Paste the article text here…")
        analyze = st.button("⚡ ANALYZE ARTICLE", use_container_width=True)
        st.caption("Tip: use a full paragraph where possible. Short headlines contain less information.")
    with right:
        st.markdown('<div class="section-label">02 / ANALYSIS PIPELINE</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="scan-console">
          <div class="radar">✦</div>
          <div class="eyebrow" style="margin-top:12px">NEURAL SCAN CORE / ONLINE</div>
          <h3 style="margin:8px 0 4px">Signal processing</h3>
          <p>01 &nbsp; Text normalization</p>
          <p>02 &nbsp; TF-IDF feature extraction</p>
          <p>03 &nbsp; Logistic regression inference</p>
          <p>04 &nbsp; Feature contribution summary</p>
          <hr>
          <div class="small-muted">The model learns statistical patterns from labelled text. It does not independently check sources, dates, images, or external evidence.</div>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        st.markdown('<div class="panel"><div class="section-label">ACTIVE DATASET</div><h4 style="margin-bottom:4px">' + html.escape(dataset_source) + '</h4><div class="small-muted">' + str(len(df)) + ' usable labelled text examples</div></div>', unsafe_allow_html=True)

    if "history" not in st.session_state:
        st.session_state.history = []
    if analyze:
        if len(article.strip()) < 20:
            st.warning("Please paste at least 20 characters so the model has some text to analyze.")
        else:
            pred = model.predict([article])[0]
            probs = model.predict_proba([article])[0]
            classes = list(model.named_steps["clf"].classes_)
            prob_map = {c: float(p) for c, p in zip(classes, probs)}
            predicted_prob = prob_map[pred]
            evidence = explain_words(model, article, pred)
            st.toast("Scan complete · Pattern analysis finished", icon="🛰️")
            st.divider()
            st.markdown('<div class="section-label">03 / SCAN REPORT</div>', unsafe_allow_html=True)
            a, b = st.columns([1, 1], gap="large")
            with a:
                label_color = "#ff7087" if pred == "FAKE" else "#54e7ff"
                title = "Potentially misleading pattern detected" if pred == "FAKE" else "No strong fake-news pattern detected"
                # The animated alert is triggered only by the model's FAKE class.
                # It is deliberately worded as a warning signal, not a factual verdict.
                if pred == "FAKE":
                    st.markdown(f'''
                    <div class="fake-alert" role="alert" aria-live="polite">
                      <div class="alert-icon">⚠</div>
                      <div>
                        <div class="alert-kicker">TRUTHLENS / CAUTION SIGNAL</div>
                        <div class="alert-title">Potential misinformation pattern detected</div>
                        <div class="alert-copy">The text resembles patterns learned from the labelled dataset. Pause before sharing and verify the claim with credible, independent sources.</div>
                      </div>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown('''
                    <div class="real-alert" role="status">
                      <div class="alert-icon">✦</div>
                      <div>
                        <div class="alert-kicker">TRUTHLENS / SCAN COMPLETE</div>
                        <div class="alert-title" style="color:#dffcff">No strong fake-news pattern detected</div>
                        <div class="alert-copy">This is not proof that the story is true. Check the original source and corroborate important claims.</div>
                      </div>
                    </div>
                    ''', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="result-box">
                  <div class="eyebrow">◉ MODEL CLASSIFICATION / SCAN COMPLETE</div>
                  <div style="margin:14px 0 8px"><span class="result-stamp" style="color:{label_color}">{'⚠ SIGNAL FLAGGED' if pred == 'FAKE' else '✦ PATTERN MATCH'}</span></div>
                  <h2 style="color:{label_color};margin:12px 0">{html.escape(title)}</h2>
                  <div style="font-size:2.5rem;font-weight:800;color:#f3f7ff">{predicted_prob*100:.1f}%</div>
                  <div class="small-muted">Model probability for the predicted class — not the probability that the story is actually true or false.</div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(predicted_prob)
            with b:
                st.markdown('<div class="panel"><div class="section-label">CLASS PROBABILITIES</div>', unsafe_allow_html=True)
                for cls in classes:
                    st.write(f"**{cls}** · {prob_map[cls]*100:.1f}%")
                    st.progress(prob_map[cls])
                st.markdown("</div>", unsafe_allow_html=True)
            st.subheader("Words that influenced this prediction")
            if evidence:
                st.caption("These are model feature contributions, not proof of misinformation.")
                st.write("  ·  ".join([f"`{word}`" for word, score in evidence]))
            else:
                st.info("No strong matching words from the model vocabulary were found in this text.")
            st.markdown('<div class="warning-box">RESPONSIBLE USE: Treat this as a screening signal, not a fact-check. Verify important claims using multiple credible sources and original documents.</div>', unsafe_allow_html=True)
            st.session_state.history.insert(0, {
                "time": datetime.now().strftime("%H:%M:%S"),
                "prediction": pred,
                "score": round(predicted_prob * 100, 1),
                "text": article[:120].replace("\n", " ")
            })
            st.subheader("Session investigation log")
            st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)
    elif "history" in st.session_state and st.session_state.history:
        st.divider()
        st.subheader("Session investigation log")
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)

elif page == "Model Lab":
    st.markdown('<div class="section-label">MODEL LAB / EVALUATION</div>', unsafe_allow_html=True)
    st.header("Inside the classifier")
    st.write("The pipeline converts text into TF-IDF features and uses logistic regression to classify the examples. Evaluation metrics below are computed from the dataset currently loaded.")
    if metrics["split"]:
        st.success(f"Independent holdout split used: {metrics['test_size']} test examples (75/25 train-test split, stratified where possible).")
    else:
        st.warning("No independent holdout split was possible; the current metrics are not a fair estimate of generalization.")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
    c2.metric("Precision (FAKE)", f"{metrics['precision']:.2f}")
    c3.metric("Recall (FAKE)", f"{metrics['recall']:.2f}")
    c4.metric("F1 (FAKE)", f"{metrics['f1']:.2f}")
    cm = metrics["matrix"]
    cm_df = pd.DataFrame(cm, index=["Actual REAL","Actual FAKE"], columns=["Predicted REAL","Predicted FAKE"])
    st.subheader("Confusion matrix")
    st.dataframe(cm_df, use_container_width=True)
    label_counts = df["label"].value_counts().rename_axis("Class").reset_index(name="Examples")
    fig = px.bar(label_counts, x="Class", y="Examples", color="Class", title="Dataset class balance", template="plotly_dark")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Dataset preview")
    st.dataframe(df.head(12), use_container_width=True, hide_index=True)
    st.caption(f"Dataset source: {dataset_source}. Use a larger, well-labelled dataset and avoid duplicate/leaked examples for a meaningful academic evaluation.")

else:
    st.markdown('<div class="section-label">ABOUT / PROJECT BRIEF</div>', unsafe_allow_html=True)
    st.header("What is TRUTHLENS AI?")
    st.write("TRUTHLENS AI is a student news-intelligence prototype. Its classifier identifies text patterns, while the Live News section discovers recent reports through RSS. RSS discovery is not the same as evidence-based verification.")
    st.subheader("Technology stack")
    st.markdown("- **TF-IDF:** converts text into numerical features based on word and phrase importance.\n- **Logistic Regression:** learns a decision boundary from labelled examples.\n- **scikit-learn Pipeline:** keeps text transformation and classification together.\n- **Streamlit:** builds the interactive web interface in Python.\n- **Plotly:** visualizes dataset balance.")
    st.subheader("Known limitations")
    st.markdown("- The demo examples are intentionally small and must not be used to judge real news.\n- Real-world reliability depends on the quality, size, balance, and date of the dataset.\n- The model can learn writing style and dataset bias instead of factual truth.\n- It does not check live sources or independently verify claims.")
    st.info("Next step: add a dedicated evidence-comparison service or fact-checking API to assess claims against multiple sources. Always inspect original reports and dates.")

st.divider()
st.markdown('<div class="small-muted" style="text-align:center">TRUTHLENS AI · Student research prototype · Always verify consequential claims independently.</div>', unsafe_allow_html=True)
