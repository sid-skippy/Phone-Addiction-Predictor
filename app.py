import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.graph_objects as go


# ---------- Load ----------
model         = joblib.load("model.pkl")
features      = joblib.load("features.pkl")
importance_df = joblib.load("importance.pkl")

st.set_page_config(layout="wide", page_title="Phone Addiction Predictor", page_icon="📱")

# ---------- SESSION STATE ----------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------- CSS ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg:      #0b0d12;
    --surface: #13161f;
    --card:    #181c27;
    --border:  #252a38;
    --accent:  #00e5a0;
    --accent2: #ff4d6d;
    --accent3: #6e8fff;
    --text:    #e8eaf0;
    --muted:   #6b7280;
    --mono:    'Space Mono', monospace;
    --sans:    'Syne', sans-serif;
}
html, body, [class*="css"] {
    font-family: var(--sans);
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
.stApp { background-color: var(--bg) !important; }
.main .block-container { max-width: 1200px; padding: 2.5rem 2rem 4rem; }
#MainMenu, footer, header { visibility: hidden; }

/* Hero */
.hero-banner {
    background: linear-gradient(135deg, #13161f 0%, #0b1020 100%);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 12px;
    padding: 2.2rem 2.5rem 1.8rem;
    margin-bottom: 2.2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(0,229,160,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: var(--sans); font-weight: 800; font-size: 2.4rem;
    letter-spacing: -0.5px; color: var(--text); margin: 0 0 0.4rem; line-height: 1.15;
}
.hero-title span { color: var(--accent); }
.hero-sub {
    font-family: var(--mono); font-size: 0.78rem;
    color: var(--muted); letter-spacing: 0.08em; text-transform: uppercase;
}
.hero-badge {
    display: inline-block;
    background: rgba(0,229,160,0.10); border: 1px solid rgba(0,229,160,0.3);
    color: var(--accent); font-family: var(--mono); font-size: 0.68rem;
    letter-spacing: 0.12em; padding: 0.25rem 0.65rem;
    border-radius: 4px; margin-bottom: 1rem; text-transform: uppercase;
}

/* Section label */
.section-label {
    font-family: var(--mono); font-size: 0.68rem; letter-spacing: 0.14em;
    color: var(--accent); text-transform: uppercase; margin-bottom: 1.1rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.section-label::after { content: ''; flex: 1; height: 1px; background: var(--border); }

/* Widgets */
.stNumberInput > div > div > input, .stSelectbox > div > div {
    background: var(--surface) !important; border: 1px solid var(--border) !important;
    border-radius: 8px !important; color: var(--text) !important;
    font-family: var(--mono) !important; font-size: 0.88rem !important;
}
/* Fix selectbox text visibility */
.stSelectbox [data-baseweb="select"] > div,
.stSelectbox [data-baseweb="select"] span,
.stSelectbox [data-baseweb="select"] div[class*="singleValue"],
.stSelectbox [data-baseweb="select"] div[class*="placeholder"],
.stSelectbox [data-baseweb="select"] input {
    color: var(--text) !important;
    background: transparent !important;
    font-family: var(--mono) !important;
    font-size: 0.88rem !important;
}
/* Dropdown list */
[data-baseweb="popover"],
[data-baseweb="popover"] *,
[data-baseweb="menu"],
[data-baseweb="menu"] *,
ul[data-baseweb="menu"],
ul[data-baseweb="menu"] li,
[role="listbox"],
[role="listbox"] *,
[role="option"] {
    background: var(--card) !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.85rem !important;
    border-color: var(--border) !important;
}
[data-baseweb="menu"] [role="option"]:hover,
[data-baseweb="menu"] [aria-selected="true"],
[role="listbox"] [role="option"]:hover,
[role="option"][aria-selected="true"] {
    background: var(--border) !important;
    color: var(--accent) !important;
}
/* Slider track */
.stSlider > div > div > div {
    background: var(--border) !important;
    height: 4px !important;
    border-radius: 4px !important;
}
/* Filled portion of track */
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, var(--accent), #00c4ff) !important;
    height: 4px !important;
    border-radius: 4px !important;
}
/* Thumb dot */
.stSlider > div > div > div > div > div {
    width: 20px !important;
    height: 20px !important;
    background: var(--accent) !important;
    border: 3px solid #0b0d12 !important;
    border-radius: 50% !important;
    box-shadow: 0 0 0 4px rgba(0,229,160,0.2), 0 0 14px rgba(0,229,160,0.5) !important;
    animation: pulse-glow 2.5s ease-in-out infinite !important;
    will-change: box-shadow !important;
}
.stSlider > div > div > div > div > div:hover,
.stSlider > div > div > div > div > div:active {
    box-shadow: 0 0 0 6px rgba(0,229,160,0.3), 0 0 28px rgba(0,229,160,0.8) !important;
    animation: none !important;
}
@keyframes pulse-glow {
    0%   { box-shadow: 0 0 0 3px rgba(0,229,160,0.15), 0 0 10px rgba(0,229,160,0.35); }
    50%  { box-shadow: 0 0 0 6px rgba(0,229,160,0.08), 0 0 20px rgba(0,229,160,0.6); }
    100% { box-shadow: 0 0 0 3px rgba(0,229,160,0.15), 0 0 10px rgba(0,229,160,0.35); }
}

/* Radio buttons as pills */
div[data-testid="stRadio"] > div { gap: 0.3rem !important; flex-wrap: wrap; justify-content: center !important; }
div[data-testid="stRadio"] > label { text-align: center !important; }
div[data-testid="stRadio"] label {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 5px !important;
    padding: 0.18rem 0.55rem !important;
    font-family: var(--mono) !important;
    font-size: 0.68rem !important;
    color: var(--muted) !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
}
div[data-testid="stRadio"] label:has(input:checked) {
    background: rgba(0,229,160,0.12) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}
div[data-testid="stRadio"] input[type="radio"],
div[data-testid="stRadio"] label svg,
div[data-testid="stRadio"] label > div:first-child { display: none !important; }

[data-testid="stTickBarMin"],
[data-testid="stTickBarMax"] {
    font-family: var(--mono) !important;
    font-size: 0.65rem !important;
    color: var(--muted) !important;
    opacity: 0.55 !important;
    letter-spacing: 0.06em !important;
    display: block !important;
    visibility: visible !important;
}
/* Hide only the hover tooltip / thumb value popup */
[data-testid="stThumbValue"],
[data-baseweb="tooltip"],
[role="tooltip"],
div[class*="Tooltip"],
div[class*="tooltip"],
div[class*="thumbValue"],
div[class*="ThumbValue"] {
    display: none !important;
    opacity: 0 !important;
    pointer-events: none !important;
}
label[data-testid="stWidgetLabel"] > div {
    font-family: var(--sans) !important; font-weight: 600 !important;
    font-size: 0.82rem !important; color: #a0a8bf !important; letter-spacing: 0.02em;
}

/* Weak tag */
.weak-tag {
    display: block;
    background: rgba(110,143,255,0.12); border: 1px solid rgba(110,143,255,0.3);
    color: var(--accent3); font-family: var(--mono); font-size: 0.62rem;
    letter-spacing: 0.1em; padding: 0.15rem 0.5rem; border-radius: 4px;
    text-transform: uppercase; margin-bottom: 0.35rem; margin-top: 1.2rem;
}

/* Button */
.stButton > button {
    background: var(--accent) !important; color: #06130e !important;
    font-family: var(--sans) !important; font-weight: 700 !important;
    font-size: 0.95rem !important; letter-spacing: 0.06em;
    border: none !important; border-radius: 8px !important;
    padding: 0.75rem 2.5rem !important; width: 100%;
    transition: all 0.18s ease !important; text-transform: uppercase;
}
.stButton > button:hover {
    background: #00ffb3 !important;
    box-shadow: 0 0 20px rgba(0,229,160,0.35) !important; transform: translateY(-1px);
}

/* Result cards */
.result-grid { display: flex; gap: 1.2rem; margin: 1.5rem 0; }
.result-card {
    flex: 1; background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.8rem 1.5rem; text-align: center;
}
.result-card.score      { border-top: 3px solid var(--accent3); }
.result-card.level-high { border-top: 3px solid var(--accent2); }
.result-card.level-med  { border-top: 3px solid #f59e0b; }
.result-card.level-low  { border-top: 3px solid var(--accent); }
.result-label {
    font-family: var(--mono); font-size: 0.68rem; letter-spacing: 0.14em;
    color: var(--muted); text-transform: uppercase; margin-bottom: 0.6rem;
}
.result-value { font-family: var(--sans); font-weight: 800; font-size: 3rem; line-height: 1; color: var(--text); }
.result-value.high   { color: var(--accent2); }
.result-value.medium { color: #f59e0b; }
.result-value.low    { color: var(--accent); }

/* Alert */
.alert-box { border-radius: 8px; padding: 1rem 1.3rem; font-family: var(--sans); font-size: 0.9rem; margin: 1rem 0; }
.alert-high { background: rgba(255,77,109,0.10); border: 1px solid rgba(255,77,109,0.35); color: #ff8fa3; }
.alert-med  { background: rgba(245,158,11,0.10); border: 1px solid rgba(245,158,11,0.35); color: #fcd34d; }
.alert-low  { background: rgba(0,229,160,0.10);  border: 1px solid rgba(0,229,160,0.35);  color: var(--accent); }

/* Tips */
.tips-grid { display: flex; flex-direction: column; gap: 0.75rem; margin-top: 0.5rem; }
.tip-card {
    background: var(--card); border: 1px solid var(--border); border-radius: 10px;
    padding: 1rem 1.2rem; display: flex; align-items: flex-start; gap: 1rem;
}
.tip-icon { font-size: 1.3rem; line-height: 1; min-width: 1.5rem; }
.tip-title { font-family: var(--sans); font-weight: 700; font-size: 0.88rem; color: var(--text); margin-bottom: 0.2rem; }
.tip-body  { font-family: var(--sans); font-size: 0.82rem; color: var(--muted); line-height: 1.5; }

/* History table */
.stDataFrame { border: 1px solid var(--border) !important; border-radius: 8px; overflow: hidden; }

hr { border-color: var(--border) !important; margin: 2rem 0 !important; }
.footer {
    font-family: var(--mono); font-size: 0.72rem; color: var(--muted);
    text-align: center; padding-top: 1.5rem; letter-spacing: 0.06em;
}
.footer span { color: var(--accent); }
</style>
""", unsafe_allow_html=True)

# ---------- HELPER FUNCTIONS (defined before any call site) ----------

def get_level(score):
    if score <= 4:     return "Low"
    elif score <= 6.5: return "Medium"
    return "High"


def make_radar(daily_usage, phone_checks, social_media, gaming, sleep, exercise, anxiety, academic_perf, social_interact):
    labels = ["Daily Usage", "Phone Checks", "Social Media", "Gaming", "Sleep*", "Exercise*", "Anxiety", "Academic*", "Social Int."]

    user_vals = [
        daily_usage  / 24,
        phone_checks / 500,
        social_media / 24,
        gaming       / 24,
        1 - (sleep    / 12),
        1 - (exercise / 10),
        (anxiety - 1) / 9,
        1 - ((academic_perf - 50) / 50),
        1 - (social_interact / 10),
    ]
    healthy_vals = [
        2.0  / 24,
        30   / 500,
        1.0  / 24,
        0.5  / 24,
        1 - (8.0 / 12),
        1 - (1.0 / 10),
        (3 - 1) / 9,
        1 - ((80 - 50) / 50),
        1 - (6 / 10),
    ]

    user_vals    = [max(0, min(1, v)) for v in user_vals]
    healthy_vals = [max(0, min(1, v)) for v in healthy_vals]

    labels       += [labels[0]]
    user_vals    += [user_vals[0]]
    healthy_vals += [healthy_vals[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=healthy_vals, theta=labels,
        fill="toself",
        fillcolor="rgba(0,229,160,0.08)",
        line=dict(color="#00e5a0", width=1.5, dash="dot"),
        name="Healthy Benchmark",
    ))
    fig.add_trace(go.Scatterpolar(
        r=user_vals, theta=labels,
        fill="toself",
        fillcolor="rgba(110,143,255,0.12)",
        line=dict(color="#6e8fff", width=2),
        name="Your Profile",
    ))
    fig.update_layout(
        height=320,
        margin=dict(t=40, b=20, l=40, r=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 1],
                showticklabels=False, gridcolor="#252a38", linecolor="#252a38",
            ),
            angularaxis=dict(
                gridcolor="#252a38", linecolor="#252a38",
                tickfont=dict(family="Space Mono", size=10, color="#a0a8bf"),
            ),
        ),
        legend=dict(
            font=dict(family="Space Mono", size=10, color="#6b7280"),
            bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)",
            orientation="h", x=0.5, xanchor="center", y=-0.08,
        ),
        font=dict(color="#e8eaf0"),
    )
    return fig


def get_tips(daily_usage, phone_checks, social_media, gaming, sleep, exercise, anxiety, academic_perf, social_interact, level):
    tips = []

    if daily_usage >= 6:
        tips.append(("Screen Time", "Your daily screen time is significantly above the recommended 2-3 hours. Try setting app timers and scheduling one phone-free hour before bed."))
    if phone_checks >= 80:
        tips.append(("Compulsive Checking", f"Checking your phone {phone_checks} times a day suggests habitual behaviour. Disable non-essential notifications and batch your checks to set intervals."))
    if social_media >= 3:
        tips.append(("Social Media", f"Spending {social_media} hrs on social media daily can amplify anxiety and FOMO. Unfollow accounts that don't add value and use grayscale mode."))
    if gaming >= 3:
        tips.append(("Gaming", f"{gaming} hrs of daily gaming is high. Set hard session limits and replace one session per day with a physical activity."))
    if sleep < 6:
        tips.append(("Sleep Deficit", f"Only {sleep} hrs of sleep is well below the 7-9 hr recommendation. Phone use before bed suppresses melatonin — try a 30-min pre-sleep screen ban."))
    if exercise < 0.5:
        tips.append(("Exercise", "Little to no daily exercise is linked to higher addictive behaviour. Even a 20-min walk improves mood and reduces compulsive phone use."))
    if anxiety >= 7:
        tips.append(("Anxiety", f"An anxiety level of {anxiety}/10 is high. Phone use often acts as avoidance coping. Try box-breathing or journaling as a replacement habit."))
    if academic_perf < 65:
        tips.append(("Academic Performance", f"A score of {academic_perf}/100 suggests phone use may be impacting your studies. Try the Pomodoro technique — 25 min focused work, 5 min break — with your phone in another room."))
    if social_interact <= 2:
        tips.append(("Social Isolation", f"Only {social_interact} social interactions per day is low. Heavy phone use can replace real-world connection. Schedule one in-person activity per day."))

    if not tips and level != "Low":
        tips.append(("General Wellness", "Your individual metrics look reasonable but the combined pattern suggests elevated risk. Focus on consistent sleep and intentional phone use."))
    if not tips:
        tips.append(("Keep it up", "Your habits are healthy. Maintain your current routine, stay mindful of gradual creep in screen time, and keep prioritising sleep and exercise."))

    return tips[:4]


# ---------- HERO ----------
st.markdown("""
<div class="hero-banner">
    <div style="display:flex; justify-content:space-between; align-items:center; height:100%;">
        <div>
            <div class="hero-badge">ML PREDICTOR &nbsp;·&nbsp; FDS PROJECT</div>
            <div class="hero-title">Phone Addiction<br><span>Predictor</span></div>
            <div class="hero-sub">Enter your usage patterns below &nbsp;--&gt;&nbsp; Get instant analysis</div>
        </div>
        <div style="text-align:right; flex-shrink:0; padding-left:2rem;">
            <div style="font-family:'Space Mono',monospace; font-size:0.6rem; letter-spacing:0.14em; color:#3d4a5c; text-transform:uppercase; margin-bottom:0.75rem;">Made by</div>
            <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:0.95rem; color:#e8eaf0; line-height:1.6;">Siddhartha Gupta</div>
            <div style="font-family:'Space Mono',monospace; font-size:0.62rem; color:#00e5a0; letter-spacing:0.08em; margin-bottom:0.6rem;">24BCE5063</div>
            <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:0.95rem; color:#e8eaf0; line-height:1.6;">Sourja Bose</div>
            <div style="font-family:'Space Mono',monospace; font-size:0.62rem; color:#00e5a0; letter-spacing:0.08em;">24BCE5110</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- INPUTS ----------
st.markdown('<div class="section-label">INPUT PARAMETERS</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    daily_usage     = st.slider("Daily Usage Hours",                          0.0, 24.0, 4.0,  step=0.25)
    phone_checks    = st.slider("Phone Checks Per Day",                       0,   500,  50,   step=5)
    anxiety         = st.slider("Anxiety Level (1=low, 10=high)",             1,   10,   5)
    academic_perf   = st.slider("Academic Performance (50=low, 100=high)",    50,  100,  75)
    social_interact = st.slider("Social Interactions Per Day (0=low, 10=high)", 0, 10,   5)

with col2:
    sleep    = st.slider("Sleep Hours",    0.0, 24.0, 7.0, step=0.25)
    exercise = st.slider("Exercise Hours", 0.0, 10.0, 1.0, step=0.25)

    st.markdown('<span class="weak-tag">WEAK PREDICTOR — low model influence</span>', unsafe_allow_html=True)
    usage_purpose = st.radio("Usage Purpose", ["Social Media", "Gaming", "Education", "Communication", "Other"], horizontal=True)

    st.markdown('<span class="weak-tag">WEAK PREDICTOR — low model influence</span>', unsafe_allow_html=True)
    parental_control = st.radio("Parental Control", ["Low", "Medium", "High"], horizontal=True)

# --- Screen time budget (social media + gaming capped by daily usage) ---
st.markdown('<div class="section-label" style="margin-top:0.5rem;">SCREEN TIME BUDGET</div>', unsafe_allow_html=True)

sm_max       = float(daily_usage)
sm_val       = float(min(st.session_state.get("sm_prev", 2.0), sm_max))
social_media = st.slider(f"Social Media Time (hrs)   ·   limit {sm_max:.2f} hrs", 0.0, max(sm_max, 0.25), sm_val, step=0.25)
st.session_state["sm_prev"] = social_media

gm_max  = max(daily_usage - social_media, 0.0)
gm_val  = float(min(st.session_state.get("gm_prev", 1.0), gm_max))
gaming  = st.slider(f"Gaming Time (hrs)   ·   limit {gm_max:.2f} hrs remaining", 0.0, max(gm_max, 0.25), gm_val, step=0.25)
gaming  = min(gaming, gm_max)
st.session_state["gm_prev"] = gaming

used    = social_media + gaming
left    = max(daily_usage - used, 0.0)
pct     = int(used / daily_usage * 100) if daily_usage > 0 else 0
bar_col = "#00e5a0" if pct <= 70 else ("#f59e0b" if pct <= 95 else "#ff4d6d")
st.markdown(f"""
<div style="margin-top:0.4rem; margin-bottom:1.8rem;">
    <div style="display:flex; justify-content:space-between; align-items:center;
                font-family:'Space Mono',monospace; font-size:0.62rem;
                color:#6b7280; letter-spacing:0.06em; margin-bottom:0.5rem;">
        <span>USED &nbsp;<span style="color:{bar_col}; font-weight:700;">{used:.2f}</span> / {daily_usage:.2f} hrs</span>
        <span style="color:{bar_col};">{left:.2f} hrs free</span>
    </div>
    <div style="background:#252a38; border-radius:6px; height:6px; overflow:hidden;">
        <div style="width:{min(pct,100)}%; height:100%; background:{bar_col};
                    border-radius:6px; transition:width 0.3s ease;"></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- JS: style tick labels + nuke hover thumb tooltip ----------
st.markdown("""
<script>
(function() {
    var style = document.createElement('style');
    style.textContent = `
        [data-testid="stThumbValue"],
        [data-testid="stSlider"] [data-baseweb="tooltip"],
        [data-testid="stSlider"] [role="tooltip"],
        [data-testid="stSlider"] div[class*="Tooltip"],
        [data-testid="stSlider"] div[class*="tooltip"],
        [data-testid="stSlider"] div[class*="thumbValue"],
        [data-testid="stSlider"] div[class*="ThumbValue"] {
            display: none !important;
            visibility: hidden !important;
            opacity: 0 !important;
            pointer-events: none !important;
        }
        [data-testid="stTickBarMin"],
        [data-testid="stTickBarMax"] {
            display: block !important;
            visibility: visible !important;
            opacity: 0.45 !important;
            font-family: 'Space Mono', monospace !important;
            font-size: 0.62rem !important;
            color: #6b7280 !important;
            letter-spacing: 0.06em !important;
        }
    `;
    document.head.appendChild(style);

    function hideThumbTooltips() {
        var thumbVals = document.querySelectorAll('[data-testid="stThumbValue"]');
        thumbVals.forEach(function(el) {
            el.style.cssText = 'display:none!important;visibility:hidden!important;opacity:0!important;';
        });
        document.querySelectorAll('[data-testid="stSlider"] div').forEach(function(el) {
            var style = window.getComputedStyle(el);
            if (style.position === 'absolute' && parseInt(style.bottom) > 10) {
                el.style.cssText = 'display:none!important;';
            }
        });
        document.querySelectorAll('[data-baseweb="popover"]').forEach(function(portal) {
            if (!portal.querySelector('style.dd-fix')) {
                var s = document.createElement('style');
                s.className = 'dd-fix';
                s.textContent = `
                    * { color: #e8eaf0 !important; background-color: #181c27 !important; }
                    [role="option"]:hover, [role="option"][aria-selected="true"] {
                        background-color: #252a38 !important;
                        color: #00e5a0 !important;
                    }
                    [role="option"][aria-selected="true"] * { color: #00e5a0 !important; }
                `;
                portal.appendChild(s);
            }
        });
    }

    hideThumbTooltips();
    var observer = new MutationObserver(hideThumbTooltips);
    observer.observe(document.body, { childList: true, subtree: true, attributes: true });
})();
</script>
""", unsafe_allow_html=True)

# ---------- build_input (needs slider variables, so stays here) ----------
def build_input():
    input_df = pd.DataFrame(columns=features)
    input_df.loc[0] = 0
    input_df.at[0, "Daily_Usage_Hours"]    = daily_usage
    input_df.at[0, "Phone_Checks_Per_Day"] = phone_checks
    input_df.at[0, "Time_on_Social_Media"] = social_media
    input_df.at[0, "Time_on_Gaming"]       = gaming
    input_df.at[0, "Sleep_Hours"]          = sleep
    input_df.at[0, "Exercise_Hours"]       = 10.0 - exercise  # inverted: more exercise → lower risk
    input_df.at[0, "Anxiety_Level"]        = anxiety
    input_df.at[0, "Academic_Performance"] = academic_perf
    input_df.at[0, "Social_Interactions"]  = social_interact
    purpose_col = f"Phone_Usage_Purpose_{usage_purpose}"
    control_col = f"Parental_Control_{parental_control}"
    if purpose_col in input_df.columns: input_df.at[0, purpose_col] = 1
    if control_col in input_df.columns: input_df.at[0, control_col] = 1
    return input_df

# ---------- PREDICT BUTTON ----------
st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("RUN PREDICTION")

if predict_btn:
    input_df = build_input()
    score    = float(model.predict(input_df)[0])
    level    = get_level(score)

    level_class = level.lower()
    level_card  = {"High": "level-high", "Medium": "level-med", "Low": "level-low"}[level]

    # Save to history
    st.session_state.history.append({
        "Run":          len(st.session_state.history) + 1,
        "Score":        round(float(score), 2),
        "Level":        level,
        "Daily Hrs":    daily_usage,
        "Checks/Day":   phone_checks,
        "Social (hrs)": social_media,
        "Gaming (hrs)": gaming,
        "Sleep (hrs)":  sleep,
        "Exercise":     exercise,
        "Anxiety":      anxiety,
        "Academic":     academic_perf,
        "Social Int.":  social_interact,
    })

    # ── SCORE CARDS ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">PREDICTION RESULTS</div>', unsafe_allow_html=True)

    st.markdown(f'''
    <div class="result-grid">
        <div class="result-card score">
            <div class="result-label">Addiction Score</div>
            <div class="result-value">{round(score, 2)}</div>
        </div>
        <div class="result-card {level_card}">
            <div class="result-label">Addiction Level</div>
            <div class="result-value {level_class}">{level}</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    if level == "High":
        st.markdown('<div class="alert-box alert-high"><strong>High addiction detected.</strong> Significantly reduce screen time, establish phone-free hours, and prioritize consistent sleep.</div>', unsafe_allow_html=True)
    elif level == "Medium":
        st.markdown('<div class="alert-box alert-med"><strong>Moderate addiction.</strong> Monitor your usage patterns and consider setting daily screen-time limits.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-box alert-low"><strong>Low addiction.</strong> You are maintaining healthy digital habits. Keep it up!</div>', unsafe_allow_html=True)

    # ── RADAR + TIPS ──
    st.markdown("<br>", unsafe_allow_html=True)
    radar_col, tips_col = st.columns([1.2, 1], gap="large")

    with radar_col:
        st.markdown('<div class="section-label">YOUR PROFILE VS HEALTHY BENCHMARKS</div>', unsafe_allow_html=True)
        st.markdown('<p style="font-family:\'Space Mono\',monospace;font-size:0.68rem;color:#6b7280;margin-bottom:0.5rem;">* Sleep and Exercise are inverted — higher means more risk (i.e. too little sleep)</p>', unsafe_allow_html=True)
        st.plotly_chart(
            make_radar(daily_usage, phone_checks, social_media, gaming, sleep, exercise, anxiety, academic_perf, social_interact),
            use_container_width=True, config={"displayModeBar": False}
        )

    with tips_col:
        st.markdown('<div class="section-label">PERSONALISED RECOMMENDATIONS</div>', unsafe_allow_html=True)
        tips = get_tips(daily_usage, phone_checks, social_media, gaming, sleep, exercise, anxiety, academic_perf, social_interact, level)
        icon_map = {
            "Screen Time": "📱",   "Compulsive Checking": "🔔",
            "Social Media": "📲",  "Gaming": "🎮",
            "Sleep Deficit": "😴", "Exercise": "🏃",
            "Anxiety": "🧘",       "General Wellness": "💡",
            "Keep it up": "✅",    "Academic Performance": "📚",
            "Social Isolation": "🤝",
        }
        cards_html = '<div class="tips-grid">'
        for title, body in tips:
            icon = icon_map.get(title, "💡")
            cards_html += f"""
            <div class="tip-card">
                <div class="tip-icon">{icon}</div>
                <div>
                    <div class="tip-title">{title}</div>
                    <div class="tip-body">{body}</div>
                </div>
            </div>"""
        cards_html += "</div>"
        st.markdown(cards_html, unsafe_allow_html=True)

    # ── FEATURE IMPORTANCE ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">FEATURE IMPORTANCE</div>', unsafe_allow_html=True)
    fi_col1, fi_col2 = st.columns(2, gap="large")

    highlight_features = ["Academic_Performance", "Social_Interactions"]

    def highlight_new(row):
        if row["feature"] in highlight_features:
            return ["background-color: rgba(110,143,255,0.12); color: #6e8fff"] * len(row)
        return [""] * len(row)

    with fi_col1:
        st.markdown("**Top Influencing Features**")
        top5 = importance_df.head(5)
        st.dataframe(top5.style.apply(highlight_new, axis=1).format({"importance": "{:.4f}"}),
                     use_container_width=True, hide_index=True)
    with fi_col2:
        st.markdown("**Least Influencing Features**")
        bot5 = importance_df.tail(5)
        st.dataframe(bot5.style.apply(highlight_new, axis=1).format({"importance": "{:.4f}"}),
                     use_container_width=True, hide_index=True)

    st.markdown('<p style="font-family:\'Space Mono\',monospace;font-size:0.72rem;color:#6b7280;margin-top:0.8rem;">Categorical features are one-hot encoded and aligned with training features. Usage Purpose and Parental Control are among the weakest predictors in this model.</p>', unsafe_allow_html=True)

# ── PREDICTION HISTORY ──
if len(st.session_state.history) > 0:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">PREDICTION HISTORY</div>', unsafe_allow_html=True)

    history_df = pd.DataFrame(st.session_state.history)

    def colour_level(val):
        if val == "High":     return "color: #ff4d6d; font-weight:700"
        elif val == "Medium": return "color: #f59e0b; font-weight:700"
        return "color: #00e5a0; font-weight:700"

    def colour_score(val):
        if val > 6.5: return "color: #ff4d6d"
        elif val > 4: return "color: #f59e0b"
        return "color: #00e5a0"

    styled = (
        history_df.style
        .format({"Score": "{:.2f}"})
        .map(colour_level, subset=["Level"])
        .map(colour_score, subset=["Score"])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    if st.button("CLEAR HISTORY"):
        st.session_state.history = []
        st.rerun()

# ---------- FOOTER ----------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    Phone Addiction Predictor &nbsp;·&nbsp; <span>ML-Powered</span> &nbsp;·&nbsp; FDS Project
</div>
""", unsafe_allow_html=True)
