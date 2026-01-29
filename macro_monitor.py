import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
import os, json
import urllib.request, urllib.parse

# Plotly dashboard (optional, but recommended)
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# =========================
# CONFIG
# =========================
START = "2012-01-01"

RSI_PERIOD = 14
DMA_BUFFER = 0.995
SMOOTH_DAYS = 3
PHASE_CONFIRM_DAYS = 5

LOOKBACK_PHASE_PCT = 252  # 1 an rolling pentru % faze
DASHBOARD_LOOKBACK_DAYS = 756  # ~3 ani pentru chart (mai rapid + clar)

TICKERS = {
    "QQQ": "QQQ",
    "SPX": "^GSPC",
    "VIX": "^VIX",
    "SMH": "SMH",
    "XLE": "XLE",
}

# ---- Alerts (Telegram) ----
TELEGRAM_ENABLED = False  # pune True după ce setezi token/chat_id
TELEGRAM_BOT_TOKEN = "PASTE_TOKEN_HERE"
TELEGRAM_CHAT_ID = "PASTE_CHAT_ID_HERE"

STATE_FILE = "risk_state.json"

# ---- Logging / dashboard outputs ----
LOG_CSV = "risk_log.csv"
DASHBOARD_HTML = "docs/index.html"

# =========================
# HELPERS
# =========================
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def load_close(ticker):
    df = yf.download(ticker, start=START, auto_adjust=True, progress=False)
    return df["Close"].dropna()

def telegram_send(text: str):
    if not TELEGRAM_ENABLED:
        return
    if "PASTE_TOKEN_HERE" in TELEGRAM_BOT_TOKEN or "PASTE_CHAT_ID_HERE" in str(TELEGRAM_CHAT_ID):
        print("⚠️ Telegram not configured (token/chat_id missing).")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": TELEGRAM_CHAT_ID, "text": text}).encode()
    urllib.request.urlopen(url, data=data, timeout=20).read()

def load_state():
    if not os.path.exists(STATE_FILE):
        return {}
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_state(state: dict):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def safe_bool(x):
    return bool(x) if pd.notna(x) else False

# =========================
# LOAD DATA (aligned)
# =========================
series = {k: load_close(v) for k, v in TICKERS.items()}
df = pd.concat(series, axis=1).dropna()
df.columns = list(TICKERS.keys())

# =========================
# INDICATORS
# =========================
df["QQQ_50"]  = df["QQQ"].rolling(50).mean()
df["SPX_200"] = df["SPX"].rolling(200).mean()
df["SMH_200"] = df["SMH"].rolling(200).mean()

df["RSI_QQQ"] = rsi(df["QQQ"], RSI_PERIOD)

df["XLE_QQQ"] = df["XLE"] / df["QQQ"]
df["XLE_QQQ_50"] = df["XLE_QQQ"].rolling(50).mean()

# =========================
# CORE MACRO SCORE (old model)
# =========================
df["score_core_raw"] = 0

# QQQ vs 50DMA
df.loc[df["QQQ"] < df["QQQ_50"] * DMA_BUFFER, "score_core_raw"] += 2

# RSI zones
df.loc[df["RSI_QQQ"] < 40, "score_core_raw"] += 3
df.loc[(df["RSI_QQQ"] >= 40) & (df["RSI_QQQ"] < 50), "score_core_raw"] += 1

# SPX vs 200DMA
df.loc[df["SPX"] < df["SPX_200"] * DMA_BUFFER, "score_core_raw"] += 2

# VIX
df.loc[df["VIX"] > 25, "score_core_raw"] += 2
df.loc[(df["VIX"] > 20) & (df["VIX"] <= 25), "score_core_raw"] += 1

# Smooth core score
df["score_core"] = df["score_core_raw"].rolling(SMOOTH_DAYS).mean().round()
df["score_core"] = df["score_core"].fillna(df["score_core_raw"])

# =========================
# CORE PHASES (phase decision ONLY from core)
# =========================
def phase_from_score(s):
    if s <= 2:
        return 1
    elif s <= 4:
        return 2
    return 3

df["phase_core_raw"] = df["score_core"].apply(phase_from_score)
df["phase_core"] = (
    df["phase_core_raw"]
    .rolling(PHASE_CONFIRM_DAYS)
    .median()
    .round()
    .fillna(df["phase_core_raw"])
    .astype(int)
)

# =========================
# ROTATION CONFIRMATIONS (cannot create RED)
# =========================
df["rot_smh_weak"] = df["SMH"] < df["SMH_200"]
df["rot_energy_on"] = df["XLE_QQQ"] > df["XLE_QQQ_50"]

df["rotation_score"] = 0
df.loc[df["rot_smh_weak"], "rotation_score"] += 1
df.loc[df["rot_energy_on"], "rotation_score"] += 1

# =========================
# DISTRIBUTIONS (rolling 1y)
# =========================
recent = df.tail(LOOKBACK_PHASE_PCT)
p1 = (recent["phase_core"] == 1).mean() * 100
p2 = (recent["phase_core"] == 2).mean() * 100
p3 = (recent["phase_core"] == 3).mean() * 100

# =========================
# FLAG LOGIC
# RED only if core phase=3 OR p3 high
# YELLOW if core phase=2 OR p2 elevated OR rotation confirmations
# GREEN otherwise
# =========================
current_core_score = float(df.iloc[-1]["score_core"])
current_core_phase = int(df.iloc[-1]["phase_core"])
smh_weak = safe_bool(df.iloc[-1]["rot_smh_weak"])
energy_on = safe_bool(df.iloc[-1]["rot_energy_on"])
current_rot = int(df.iloc[-1]["rotation_score"])

reasons = []

if current_core_phase == 3 or p3 >= 20:
    flag = "RED"
    flag_text = "🔴 RED – defensiv (core macro risk-off)"
    if current_core_phase == 3:
        reasons.append("Core phase = FAZA 3")
    if p3 >= 20:
        reasons.append(f"Core FAZA 3 % (1y) = {p3:.1f}% ≥ 20%")
else:
    yellow = False
    if current_core_phase == 2:
        yellow = True
        reasons.append("Core phase = FAZA 2")
    if p2 >= 20:
        yellow = True
        reasons.append(f"Core FAZA 2 % (1y) = {p2:.1f}% ≥ 20%")
    if current_rot >= 1:
        yellow = True
        if smh_weak:
            reasons.append("SMH sub 200DMA (semis slăbiți)")
        if energy_on:
            reasons.append("XLE/QQQ peste 50DMA (rotație spre energie)")

    if yellow:
        flag = "YELLOW"
        flag_text = "🟡 YELLOW – pregătește rotația (nu panică)"
    else:
        flag = "GREEN"
        flag_text = "🟢 GREEN – risk on"
# =========================
# YELLOW TYPE (context only)
# =========================
yellow_type = ""
action_hint = ""

if flag == "YELLOW":
    # 1) Rotation-only yellow: core is phase 1 and score low, but confirmations triggered
    if (current_core_phase == 1) and (current_core_score <= 2) and (smh_weak or energy_on):
        yellow_type = "YELLOW (Rotation)"
        action_hint = (
            "Core e curat (risk-on), dar leadership-ul se mută. "
            "Nu vinzi. Doar reduci agresivitatea pe high-beta și "
            "direcționezi 10–20% din intrările noi spre energie/defensive."
        )
    # 2) Core deterioration yellow: phase 2 or rising phase-2 share
    else:
        yellow_type = "YELLOW (Core deterioration)"
        action_hint = (
            "Core-ul începe să se deterioreze. "
            "Încetinești DCA pe tech (ex: 50–70% din cash nou), "
            "eviți leverage, și aștepți confirmare (persistență 2–3 săptămâni)."
        )

elif flag == "RED":
    yellow_type = "RED (Risk-off)"
    action_hint = (
        "Regim defensiv. Reduci expunerea totală, eviți high-beta, "
        "crești cash/defensive. Execuție graduală, nu panică."
    )
else:
    yellow_type = "GREEN (Risk-on)"
    action_hint = (
        "Regim favorabil. DCA normal, buy dips selectiv. "
        "Nu supra-optimiza; respectă sizing."
    )

print(f"\n🧩 Regime label: {yellow_type}")
print(f"🧭 Action hint: {action_hint}")

# =========================
# OUTPUT
# =========================
today = str(date.today())

print("\n📊 MACRO MONITOR (CORE + ROTATION CONFIRMATIONS)")
print("------------------------------------------------")
print(f"Data: {today}")
print(f"Core score: {current_core_score}")
print(f"Core phase: FAZA {current_core_phase}")
print(f"Distribuție core (ultimele 12 luni): FAZA1 {p1:.1f}% | FAZA2 {p2:.1f}% | FAZA3 {p3:.1f}%")
print(f"Confirmări rotație: SMH<200DMA={'YES' if smh_weak else 'NO'} | XLE/QQQ>50DMA={'YES' if energy_on else 'NO'}")
print("\nVERDICT:", flag_text)
if reasons:
    print("Motiv(e):")
    for r in reasons:
        print(" -", r)

# =========================
# 1) CSV LOG (append, no duplicate date)
# =========================
log_row = {
    "date": today,
    "flag": flag,
    "core_score": current_core_score,
    "core_phase": current_core_phase,
    "p1_1y": round(p1, 2),
    "p2_1y": round(p2, 2),
    "p3_1y": round(p3, 2),
    "smh_weak": smh_weak,
    "energy_rotation": energy_on
}

if os.path.exists(LOG_CSV):
    log_df = pd.read_csv(LOG_CSV)
    # remove any existing row for today, then append
    log_df["date"] = log_df["date"].astype(str)
    log_df = log_df[log_df["date"] != str(today)]

    log_df = pd.concat([log_df, pd.DataFrame([log_row])], ignore_index=True)
else:
    log_df = pd.DataFrame([log_row])

log_df.to_csv(LOG_CSV, index=False)
print(f"\n✅ Logged to CSV: {LOG_CSV}")
# =========================
# CONTEXT CARD (extra detail, does NOT change verdict)
# =========================
last = df.iloc[-1]

qqq = float(last["QQQ"])
qqq_50 = float(last["QQQ_50"])
spx = float(last["SPX"])
spx_200 = float(last["SPX_200"])
rsi_q = float(last["RSI_QQQ"])
vix = float(last["VIX"])

# Distances to key thresholds
qqq_vs_50_pct = (qqq / qqq_50 - 1) * 100
spx_vs_200_pct = (spx / spx_200 - 1) * 100
rsi_to_50 = rsi_q - 50
rsi_to_40 = rsi_q - 40
vix_to_20 = vix - 20
vix_to_25 = vix - 25

# Component states (human-readable)
price_state = "OK" if qqq >= qqq_50 * DMA_BUFFER else "BROKEN"
spx_state = "OK" if spx >= spx_200 * DMA_BUFFER else "BROKEN"

if rsi_q < 40:
    rsi_state = "DANGER (<40)"
elif rsi_q < 50:
    rsi_state = "WEAK (40-50)"
else:
    rsi_state = "OK (>=50)"

if vix > 25:
    vix_state = "HIGH (>25)"
elif vix > 20:
    vix_state = "ELEVATED (20-25)"
else:
    vix_state = "OK (<=20)"

# Heat 0–100 (context only)
heat = 0
# Price/MA proximity
heat += 30 if price_state == "BROKEN" else (10 if qqq_vs_50_pct < 1.0 else 0)
# RSI
heat += 30 if rsi_q < 40 else (15 if rsi_q < 50 else 0)
# SPX breadth
heat += 20 if spx_state == "BROKEN" else (8 if spx_vs_200_pct < 2.0 else 0)
# VIX
heat += 20 if vix > 25 else (10 if vix > 20 else 0)
# Include rotation as mild heat (context)
if energy_on or smh_weak:
    heat += 10

# Include core phase (context)
if current_core_phase == 2:
    heat += 20
elif current_core_phase == 3:
    heat += 40
# Mild heat for rotation confirmations
if energy_on or smh_weak:
    heat += 10

# Phase-based heat (context)
if current_core_phase == 2:
    heat += 20
elif current_core_phase == 3:
    heat += 40

heat = int(max(0, min(100, heat)))

watch = []
if qqq_vs_50_pct < 1.0: watch.append("QQQ aproape de 50DMA")
if spx_vs_200_pct < 2.0: watch.append("SPX aproape de 200DMA")
if 40 <= rsi_q < 55: watch.append("RSI în zona de tranziție")
if 18 <= vix <= 22: watch.append("VIX aproape de 20")
if not watch: watch.append("Nimic critic – trend stabil")

print("\n🧾 CONTEXT CARD (nu schimbă verdictul)")
print("------------------------------------")
print(f"Heat (0-100): {heat}")
print(f"QQQ vs 50DMA: {qqq_vs_50_pct:+.2f}% | State: {price_state}")
print(f"SPX vs 200DMA: {spx_vs_200_pct:+.2f}% | State: {spx_state}")
print(f"RSI(QQQ): {rsi_q:.1f} | {rsi_state} | Δ50={rsi_to_50:+.1f} | Δ40={rsi_to_40:+.1f}")
print(f"VIX: {vix:.1f} | {vix_state} | Δ20={vix_to_20:+.1f} | Δ25={vix_to_25:+.1f}")
print("Watch:")
for w in watch:
    print(" -", w)

# Add to CSV log row
log_row.update({
    "heat": heat,
    "qqq_vs_50_pct": round(qqq_vs_50_pct, 3),
    "spx_vs_200_pct": round(spx_vs_200_pct, 3),
    "rsi": round(rsi_q, 2),
    "vix": round(vix, 2),
    "price_state": price_state,
    "spx_state": spx_state,
    "rsi_state": rsi_state,
    "vix_state": vix_state,
    "watch": " | ".join(watch),
    "regime_label": yellow_type,
    "action_hint": action_hint
})


# =========================
# 2) Plotly Dashboard (HTML)
# =========================
if PLOTLY_OK:
    df_dash = df.tail(DASHBOARD_LOOKBACK_DAYS).copy()

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.4, 0.3, 0.3],
        subplot_titles=("QQQ + 50DMA", "RSI (QQQ)", "Risk Phase")
    )

    # --- PRICE ---
    fig.add_trace(
        go.Scatter(x=df_dash.index, y=df_dash["QQQ"], name="QQQ"),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df_dash.index,
            y=df_dash["QQQ_50"] * DMA_BUFFER,
            name="50DMA (buffer)",
            line=dict(dash="dot")
        ),
        row=1, col=1
    )

    # --- RSI ---
    fig.add_trace(
        go.Scatter(x=df_dash.index, y=df_dash["RSI_QQQ"], name="RSI"),
        row=2, col=1
    )
    fig.add_hline(y=40, line_dash="dash", row=2, col=1)

    # --- PHASE ---
    fig.add_trace(
        go.Scatter(
            x=df_dash.index,
            y=df_dash["phase_core"],
            name="Phase",
            mode="lines",
            fill="tozeroy"
        ),
        row=3, col=1
    )

    # --- HEADER & CONTEXT ---
    header_line = f"{yellow_type} | Heat: {heat}/100 | Core: FAZA {current_core_phase} (score {current_core_score:.1f})"
    sub_line = f"Rotation: SMH<200DMA={'YES' if smh_weak else 'NO'} | XLE/QQQ>50DMA={'YES' if energy_on else 'NO'} | Updated: {today}"

    fig.update_layout(
        title=header_line,
        height=900,
        showlegend=True,
        margin=dict(t=120)
    )

    fig.add_annotation(
        text=sub_line,
        xref="paper", yref="paper",
        x=0, y=1.10,
        showarrow=False,
        align="left"
    )

    fig.add_annotation(
        text=f"<b>Action:</b> {action_hint}",
        xref="paper", yref="paper",
        x=0, y=1.06,
        showarrow=False,
        align="left"
    )

    fig.write_html(DASHBOARD_HTML)
    print(f"✅ Dashboard generated: {DASHBOARD_HTML}")
  

# =========================
# ALERT ON CHANGE ONLY
# =========================
state = load_state()
prev_flag = state.get("flag")

state.update({
    "date": today,
    "flag": flag,
    "core_score": current_core_score,
    "core_phase": current_core_phase,
    "p1": p1, "p2": p2, "p3": p3,
    "smh_weak": smh_weak,
    "energy_on": energy_on
})
save_state(state)

if prev_flag != flag:
    msg = (
        f"🚦 Risk flag changed: {prev_flag} → {flag}\n"
        f"{flag_text}\n"
        f"Core score: {current_core_score} | Core phase: FAZA {current_core_phase}\n"
        f"1y core phase%: P1 {p1:.1f}% | P2 {p2:.1f}% | P3 {p3:.1f}%\n"
        f"SMH<200DMA: {'YES' if smh_weak else 'NO'} | XLE/QQQ>50DMA: {'YES' if energy_on else 'NO'}"
    )
    telegram_send(msg)
    print("\n✅ Alert sent (flag changed).")
else:
    print("\nℹ️ No alert (flag unchanged).")
