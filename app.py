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


st.markdown("""
<style>
:root{--bg:#030406;--red:#ff4658;--green:#54ef8b;--line:rgba(255,255,255,.12);--muted:#a9a9b2}
html,body,[class*="css"]{font-family:'Manrope',sans-serif}
.stApp{background:radial-gradient(ellipse at 85% 18%,rgba(255,44,66,.10),transparent 28%),#030406!important;color:#f7f7f8}
[data-testid="stHeader"]{background:rgba(3,4,6,.9)}
.block-container{max-width:1500px;padding-top:1.1rem}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#060709,#09090c)!important;border-right:1px solid rgba(255,70,88,.19)}
[data-testid="stSidebar"] *{color:#f4f4f6}
[data-testid="stSidebar"] [data-testid="stRadio"] label{padding:8px 10px;border-radius:12px}
h1,h2,h3,h4{letter-spacing:-.035em;color:#fafafa}
p,li{color:#c9c9d0}
.eyebrow,.section-label{color:#ff6574!important;font-family:'DM Mono',monospace;letter-spacing:.13em;font-size:.72rem;text-transform:uppercase}
.hero,.panel,.metric-card,.live-card{background:linear-gradient(145deg,rgba(16,15,18,.98),rgba(6,7,9,.98))!important;border:1px solid var(--line)!important;border-radius:17px!important}
.hero{padding:26px!important}
.hero-title{font-size:clamp(2rem,4vw,3.3rem)!important}
.gradient{background:linear-gradient(90deg,#fff,#ff5264)!important;-webkit-background-clip:text!important;background-clip:text!important;color:transparent!important}
.pill{border-color:rgba(255,70,88,.35)!important;color:#ff9ca6!important;background:rgba(255,70,88,.07)!important}
div.stButton>button{border-radius:11px!important;border:1px solid rgba(255,75,91,.45)!important;background:linear-gradient(100deg,#241014,#121013)!important;color:#fff!important;font-weight:700;min-height:2.7rem}
div.stButton>button:hover{border-color:#ff5365!important;box-shadow:0 0 24px rgba(255,60,80,.17)!important}
.stTextArea textarea,.stTextInput input,div[data-baseweb="select"]>div{background:#090a0d!important;color:#f8f8fa!important;border:1px solid rgba(255,255,255,.15)!important;border-radius:10px!important}
hr{border-color:var(--line)!important}
.small-muted{color:#9696a1;font-size:.82rem;line-height:1.6}
.result-box{background:linear-gradient(135deg,rgba(34,12,17,.85),rgba(8,8,10,.95))!important;border-color:rgba(255,65,82,.4)!important}
.warning-box{border-color:rgba(255,174,74,.34)!important;background:rgba(255,174,74,.06)!important}
.scan-console{background:linear-gradient(145deg,#110b0e,#070709)!important;border-color:rgba(255,70,88,.25)!important}
.scan-console:after{background:linear-gradient(180deg,transparent,rgba(255,65,82,.12),transparent)!important}
.radar{border-color:rgba(255,83,100,.7)!important;color:#ff6877!important}
.fake-alert{animation:flickerIn .55s ease-out both,pulseGlowRed 2.4s ease-in-out .5s infinite!important}
.real-alert{border-color:rgba(62,220,130,.4)!important;background:linear-gradient(120deg,rgba(7,39,22,.85),rgba(7,10,9,.95))!important}
@keyframes pulseGlowRed{0%,100%{box-shadow:0 0 12px rgba(255,55,75,.08)}50%{box-shadow:0 0 30px rgba(255,55,75,.25)}}
.live-card{padding:16px 18px;margin:10px 0}
.live-card a{color:#ff8791;text-decoration:none}
.live-meta{font:.7rem 'DM Mono',monospace;color:#9999a4;letter-spacing:.06em;text-transform:uppercase}
.verdict-fake{border:1px solid rgba(255,45,65,.65);background:linear-gradient(100deg,rgba(54,10,18,.92),rgba(16,8,11,.9));box-shadow:0 0 35px rgba(255,45,65,.10);border-radius:15px;padding:25px;animation:pulseGlowRed 2.6s infinite}
.verdict-true{border:1px solid rgba(71,235,133,.55);background:linear-gradient(100deg,rgba(7,48,27,.9),rgba(7,15,12,.92));box-shadow:0 0 30px rgba(71,235,133,.08);border-radius:15px;padding:25px}
.verdict-title{font-size:clamp(1.5rem,3vw,2.1rem);font-weight:800;letter-spacing:-.03em}
.hero-home{min-height:470px;display:grid;grid-template-columns:1.15fr .85fr;gap:14px;align-items:center;padding:25px 26px;border:1px solid rgba(255,255,255,.13);border-radius:20px;background:radial-gradient(ellipse at 85% 48%,rgba(255,40,58,.14),transparent 34%),linear-gradient(120deg,#060709,#09090d);overflow:hidden}
.hero-copybox{text-align:center;position:relative;z-index:1}
.hero-copybox h1{font-size:clamp(2.1rem,3.8vw,3.4rem);margin:8px 0}
.hero-copybox p{font-size:.95rem;max-width:480px;margin:12px auto;color:#c6c6cd}
.globe-wrap{display:flex;align-items:center;justify-content:center;min-height:300px}
.globe-svg{width:min(100%,390px);filter:drop-shadow(0 0 18px rgba(255,48,67,.24));animation:globeFloat 6s ease-in-out infinite}
@keyframes globeFloat{0%,100%{transform:translateY(0) rotate(-2deg)}50%{transform:translateY(-9px) rotate(2deg)}}
.home-card{height:115px;border:1px solid rgba(255,255,255,.15);border-radius:14px;background:linear-gradient(140deg,#111116,#08090b);padding:18px;display:flex;flex-direction:column;justify-content:center}
.home-card b{font-size:1rem;color:#fff}.home-card span{font-size:.77rem;color:#b8b8c1;margin-top:5px}
@media(max-width:850px){.hero-home{grid-template-columns:1fr}.globe-wrap{min-height:200px}.globe-svg{max-width:260px}}
</style>
""", unsafe_allow_html=True)

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
    st.markdown('<div style="font-size:2.4rem;color:#ff6372;text-align:center;line-height:1.2">△</div><div style="text-align:center;font-weight:800;letter-spacing:-.03em">TRUTHLENS <span style="color:#ff4658">AI</span></div><div style="text-align:center;font-size:.58rem;color:#aaa;margin-bottom:24px">Verify. Trust. Stay Informed.</div>', unsafe_allow_html=True)
    if "navigation_target" in st.session_state:
        st.session_state["nav_page"] = st.session_state.pop("navigation_target")
    page = st.radio("NAVIGATION", ["Home", "Scanner", "Live News", "Evidence", "Model Lab", "About"], index=0, key="nav_page", label_visibility="visible")
    st.divider()
    st.markdown('<div class="section-label">SYSTEM STATUS</div>', unsafe_allow_html=True)
    st.markdown("🟢 Interface online")
    st.markdown("🔵 Local ML pipeline")
    st.markdown("🟡 Live news: RSS discovery")
    st.divider()
    st.markdown('<div class="small-muted">REFERENCE UI · RED GLASS<br>Designed for responsible media literacy.</div>', unsafe_allow_html=True)

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
    st.markdown("""
    <div class="hero-home">
      <div class="hero-copybox">
        <div style="font-size:4rem;line-height:1;color:#f5f5f7">△</div>
        <div class="hero-title">TRUTHLENS <span class="gradient">AI</span></div>
        <div style="font-size:1.05rem;color:#fff">Verify. Trust. Stay Informed.</div>
        <p>An AI-powered platform to detect fake news, verify claims and analyze news credibility.</p>
        <div style="height:1px;background:linear-gradient(90deg,transparent,#34343b,transparent);margin:22px 0"></div>
        <div style="color:#d7d7dc;font-size:.9rem">Or explore the latest news below<br><span style="font-size:1.5rem">↓</span></div>
      </div>
      <div class="globe-wrap">
      <svg class="globe-svg" viewBox="0 0 420 420" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Glowing red digital globe">
       <defs><radialGradient id="g" cx="35%" cy="30%"><stop offset="0" stop-color="#ff8b94"/><stop offset=".38" stop-color="#b51b31"/><stop offset="1" stop-color="#09090d"/></radialGradient><pattern id="dots" width="6" height="6" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="1.25" fill="#ff5268" opacity=".9"/></pattern></defs>
       <circle cx="220" cy="205" r="128" fill="url(#g)" opacity=".55"/><circle cx="220" cy="205" r="128" fill="url(#dots)" stroke="#ff4d62" stroke-width="1" opacity=".92"/>
       <ellipse cx="220" cy="205" rx="58" ry="128" fill="none" stroke="#ff6677" opacity=".65"/><ellipse cx="220" cy="205" rx="102" ry="128" fill="none" stroke="#ff6677" opacity=".4"/><ellipse cx="220" cy="205" rx="128" ry="45" fill="none" stroke="#ff6677" opacity=".55"/><ellipse cx="220" cy="205" rx="128" ry="88" fill="none" stroke="#ff6677" opacity=".35"/>
       <path d="M132 151 Q158 130 176 147 L194 161 184 184 164 190 155 215 137 202Z M218 145 L246 137 266 157 259 176 280 193 267 217 246 211 230 190 211 177Z M245 233 L266 246 259 274 239 293 225 274 229 251Z" fill="#ff5268" opacity=".85"/>
       <g fill="none" stroke="#ff4b61" stroke-width="1.3" opacity=".9"><ellipse cx="220" cy="205" rx="158" ry="63" transform="rotate(-25 220 205)"/><ellipse cx="220" cy="205" rx="170" ry="83" transform="rotate(22 220 205)"/><path d="M54 215 Q165 55 363 160"/><path d="M73 267 Q190 340 372 221"/></g>
       <g fill="#ff7887"><circle cx="108" cy="159" r="3"/><circle cx="345" cy="139" r="2.5"/><circle cx="365" cy="247" r="3"/><circle cx="150" cy="316" r="2"/><circle cx="289" cy="72" r="2.5"/></g>
      </svg>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    c1,c2=st.columns(2,gap="large")
    with c1:
        st.markdown('<div class="home-card"><b>▤ &nbsp; Headline Verification</b><span>Check a news claim or headline</span></div>',unsafe_allow_html=True)
        if st.button("→ VERIFY A HEADLINE",use_container_width=True,key="home_headline"):
            st.session_state["navigation_target"]="Scanner"; st.rerun()
    with c2:
        st.markdown('<div class="home-card"><b>▤ &nbsp; Full Article Analysis</b><span>Paste and analyse the complete article</span></div>',unsafe_allow_html=True)
        if st.button("→ ANALYSE AN ARTICLE",use_container_width=True,key="home_article"):
            st.session_state["navigation_target"]="Scanner"; st.rerun()
    st.markdown("### Latest news discovery")
    q=st.text_input("Search recent news",value=st.session_state.get("recent_news_query","India"),key="home_news_query",placeholder="Try a topic or claim, e.g. space mission")
    if st.button("SEARCH RECENT NEWS ↗",use_container_width=True,key="home_fetch_news"):
        try:
            st.session_state["recent_news_items"]=fetch_google_news(q,5); st.session_state["recent_news_query"]=q
        except Exception:
            st.warning("Could not reach the live news feed. External RSS access may be blocked; try again later.")
    for item in st.session_state.get("recent_news_items",[])[:3]:
        st.markdown(f'<div class="live-card"><div class="live-meta">{html.escape(item["source"])} · {html.escape(item["published"])}</div><h4>{html.escape(item["title"])}</h4><a href="{html.escape(item["link"],quote=True)}" target="_blank">READ REPORT ↗</a></div>',unsafe_allow_html=True)

