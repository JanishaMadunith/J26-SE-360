"""Module 1 — Multimodal Affective State Detection (owner: J.D. Jayatilake, IT23279698).
"""
import io
import json
import joblib
import librosa
import numpy as np
import soundfile as sf
import streamlit as st

import os
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Multimodal Affective State Detection",
    page_icon="🧠",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "models", "ravdess_mfcc_random_forest_5class.pkl"))

CLASSES = ["Neutral", "Calm", "Happy", "Sad", "Angry"]

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load voice model: {e}")
    st.stop()


def predict_typing_emotion(wpm, hold_time, flight_time, jitter, error_rate):
    """
    Robust bounded behavioral scoring model based on keystroke biometric norms (Epp et al.).
    Maps typing rhythm, latency, velocity, and error correction to 5 affective states.
    Prevents outlier thinking pauses from biasing the model to 'Angry'.
    Returns softmax probabilities across CLASSES: ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry'].
    """
    wpm_c = np.clip(wpm, 10, 130)
    hold_c = np.clip(hold_time, 40, 220)
    jitter_c = np.clip(jitter, 15, 110)
    error_c = np.clip(error_rate, 0, 45)

    # 1. Happy: swift typing, crisp light hold (<90ms), low error rate
    z_wpm_h = np.clip((wpm_c - 42) / 18.0, -1.5, 2.0)
    z_hold_h = np.clip((95 - hold_c) / 25.0, -1.5, 2.0)
    z_err_h = np.clip((4 - error_c) / 5.0, -1.5, 1.0)
    s_happy = 0.5 * z_wpm_h + 0.6 * z_hold_h + 0.4 * z_err_h

    # 2. Calm: rhythmic cadence (low jitter), uniform hold time, minimal backspaces
    z_jit_c = np.clip((42 - jitter_c) / 18.0, -1.5, 2.0)
    z_err_c = np.clip((4 - error_c) / 4.0, -1.5, 1.5)
    z_pace_c = np.clip(1.4 - abs(wpm_c - 40) / 15.0, -1.0, 1.5)
    s_calm = 0.7 * z_jit_c + 0.5 * z_err_c + 0.4 * z_pace_c

    # 3. Sad: slow velocity, prolonged lingering key hold (>125ms), sluggish pauses
    z_wpm_s = np.clip((35 - wpm_c) / 12.0, -1.5, 2.0)
    z_hold_s = np.clip((hold_c - 110) / 25.0, -1.5, 2.0)
    s_sad = 0.7 * z_wpm_s + 0.7 * z_hold_s

    # 4. Angry: requires BOTH elevated error/correction rate (>12%) AND erratic jitter (>55ms)
    z_err_a = np.clip((error_c - 12) / 8.0, -1.5, 2.5)
    z_jit_a = np.clip((jitter_c - 55) / 22.0, -1.5, 2.0)
    hold_penalty = (hold_c - 130) / 25.0 if hold_c > 130 else 0.0
    s_angry = (0.75 * z_err_a + 0.65 * z_jit_a) - hold_penalty

    # 5. Neutral: baseline equilibrium when extreme affective deviations are absent
    max_dev = max(s_happy, s_calm, s_sad, s_angry)
    s_neutral = 0.3 - (0.45 * max_dev)

    raw_scores = np.array([s_neutral, s_calm, s_happy, s_sad, s_angry], dtype=float)
    exp_scores = np.exp(raw_scores - np.max(raw_scores))
    probs = exp_scores / np.sum(exp_scores)
    return probs


st.title(" Multimodal Affective State Assessment")
st.write(
    "Assess your current emotional state using multiple complementary modalities: "
    "**Subjective Self-Report**, **Behavioral Typing Dynamics**, and **Acoustic Voice Check-in**."
)

options = [
    "Not at all",
    "A little",
    "Moderately",
    "Very",
    "Extremely"
]

st.subheader("1. How happy do you feel right now?")
happy = options.index(st.radio(
    "Happy", options, horizontal=True, key="happy",
    label_visibility="collapsed"
))

st.subheader("2. How calm and relaxed do you feel?")
calm = options.index(st.radio(
    "Calm", options, horizontal=True, key="calm",
    label_visibility="collapsed"
))

st.subheader("3. How sad or down do you feel?")
sad = options.index(st.radio(
    "Sad", options, horizontal=True, key="sad",
    label_visibility="collapsed"
))

st.subheader("4. How angry or irritated do you feel?")
angry = options.index(st.radio(
    "Angry", options, horizontal=True, key="angry",
    label_visibility="collapsed"
))

st.subheader("5. How emotionally neutral do you feel?")
neutral = options.index(st.radio(
    "Neutral", options, horizontal=True, key="neutral",
    label_visibility="collapsed"
))

st.subheader("6. Which best describes your current feeling?")
reported_emotion = st.selectbox(
    "Choose your current emotion",
    ["Neutral", "Calm", "Happy", "Sad", "Angry"],
    index=None,
    placeholder="Select your current emotion (optional)..."
)

st.subheader("7. Describe how you feel (Reply by Typing - Behavioral Dynamics)")


typing_html_component = """
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
      :root {
        --bg-card: #f8f9fa;
        --bg-box: #ffffff;
        --border-color: #e2e8f0;
        --text-primary: #2d3748;
        --text-secondary: #718096;
        --text-muted: #4a5568;
        --bar-bg: #edf2f7;
        --pill-bg: #edf2f7;
        --pill-border: #cbd5e0;
        --pill-hover: #e2e8f0;
        --pill-text: #4a5568;
        --textarea-bg: #ffffff;
        --textarea-border: #cbd5e0;
        --textarea-text: #2d3748;
        --shadow: 0 1px 4px rgba(0,0,0,0.03);
      }

      @media (prefers-color-scheme: dark) {
        :root {
          --bg-card: #151821;
          --bg-box: #1d212c;
          --border-color: #2b3142;
          --text-primary: #f0f2f6;
          --text-secondary: #a0aec0;
          --text-muted: #cbd5e0;
          --bar-bg: #272c3d;
          --pill-bg: #1d212c;
          --pill-border: #2e3547;
          --pill-hover: #262c3d;
          --pill-text: #cbd5e0;
          --textarea-bg: #0e1117;
          --textarea-border: #303749;
          --textarea-text: #f0f2f6;
          --shadow: 0 2px 8px rgba(0,0,0,0.3);
        }
      }

      body.dark-theme {
        --bg-card: #151821;
        --bg-box: #1d212c;
        --border-color: #2b3142;
        --text-primary: #f0f2f6;
        --text-secondary: #a0aec0;
        --text-muted: #cbd5e0;
        --bar-bg: #272c3d;
        --pill-bg: #1d212c;
        --pill-border: #2e3547;
        --pill-hover: #262c3d;
        --pill-text: #cbd5e0;
        --textarea-bg: #0e1117;
        --textarea-border: #303749;
        --textarea-text: #f0f2f6;
        --shadow: 0 2px 8px rgba(0,0,0,0.3);
      }

      body.light-theme {
        --bg-card: #f8f9fa;
        --bg-box: #ffffff;
        --border-color: #e2e8f0;
        --text-primary: #2d3748;
        --text-secondary: #718096;
        --text-muted: #4a5568;
        --bar-bg: #edf2f7;
        --pill-bg: #edf2f7;
        --pill-border: #cbd5e0;
        --pill-hover: #e2e8f0;
        --pill-text: #4a5568;
        --textarea-bg: #ffffff;
        --textarea-border: #cbd5e0;
        --textarea-text: #2d3748;
        --shadow: 0 1px 4px rgba(0,0,0,0.03);
      }

      body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        margin: 0;
        padding: 5px 0;
        color: var(--text-primary);
        background: transparent;
        transition: color 0.3s ease;
      }
      .card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 16px;
        box-shadow: var(--shadow);
        transition: background 0.3s ease, border-color 0.3s ease;
      }
      textarea {
        width: 100%;
        box-sizing: border-box;
        height: 95px;
        padding: 12px;
        border-radius: 8px;
        border: 1.5px solid var(--textarea-border);
        font-size: 14px;
        line-height: 1.5;
        outline: none;
        resize: vertical;
        transition: border 0.2s, background 0.3s, color 0.3s;
        font-family: inherit;
        background: var(--textarea-bg);
        color: var(--textarea-text);
      }
      textarea::placeholder {
        color: var(--text-secondary);
        opacity: 0.75;
      }
      textarea:focus {
        border-color: #3182ce;
        box-shadow: 0 0 0 2px rgba(49, 130, 206, 0.25);
      }
      .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(115px, 1fr));
        gap: 10px;
        margin-top: 12px;
      }
      .stat-box {
        background: var(--bg-box);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 8px 10px;
        text-align: center;
        transition: background 0.3s ease, border-color 0.3s ease;
      }
      .stat-title {
        font-size: 10px;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
        margin-bottom: 3px;
      }
      .stat-val {
        font-size: 16px;
        font-weight: 700;
        color: var(--text-primary);
      }
      .result-banner {
        margin-top: 14px;
        padding: 12px 14px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: var(--bg-box);
        border: 1px solid var(--border-color);
        border-left: 5px solid #a0aec0;
        transition: background 0.3s ease, border-color 0.3s ease;
      }
      .top-emotion {
        font-size: 18px;
        font-weight: 800;
        display: flex;
        align-items: center;
        gap: 8px;
      }
      .prob-container {
        margin-top: 12px;
      }
      .prob-row {
        display: flex;
        align-items: center;
        margin-bottom: 6px;
        font-size: 12px;
      }
      .prob-label {
        width: 75px;
        font-weight: 600;
      }
      .prob-bar-bg {
        flex: 1;
        background: var(--bar-bg);
        height: 10px;
        border-radius: 5px;
        overflow: hidden;
        margin: 0 10px;
        transition: background 0.3s ease;
      }
      .prob-bar-fill {
        height: 100%;
        border-radius: 5px;
        transition: width 0.3s ease;
      }
      .prob-val {
        width: 45px;
        text-align: right;
        font-weight: 600;
        color: var(--text-primary);
      }
      .btn-reset {
        background: var(--pill-bg);
        border: 1px solid var(--pill-border);
        color: var(--pill-text);
        padding: 5px 12px;
        border-radius: 6px;
        cursor: pointer;
        font-size: 11px;
        font-weight: 600;
        transition: all 0.2s;
      }
      .btn-reset:hover {
        background: var(--pill-hover);
      }
      .diagnosis {
        margin-top: 10px;
        font-size: 12px;
        color: var(--text-muted);
        background: var(--bg-box);
        border: 1px solid var(--border-color);
        padding: 8px 12px;
        border-radius: 6px;
        line-height: 1.4;
        transition: background 0.3s ease, color 0.3s ease;
      }
    </style>
    </head>
    <body>

    <div class="card">
      <textarea id="typingArea" placeholder="Write a few words about what might be affecting your mood..."></textarea>

      <div class="stats-grid">
        <div class="stat-box">
          <div class="stat-title">⚡ Speed</div>
          <div class="stat-val" id="statWpm">0 <span style="font-size:11px; font-weight:normal;">WPM</span></div>
        </div>
        <div class="stat-box">
          <div class="stat-title">⏱️ Hold Time</div>
          <div class="stat-val" id="statHold">0 <span style="font-size:11px; font-weight:normal;">ms</span></div>
        </div>
        <div class="stat-box">
          <div class="stat-title">🔄 Flight Time</div>
          <div class="stat-val" id="statFlight">0 <span style="font-size:11px; font-weight:normal;">ms</span></div>
        </div>
        <div class="stat-box">
          <div class="stat-title">📈 Rhythm Jitter</div>
          <div class="stat-val" id="statJitter">0 <span style="font-size:11px; font-weight:normal;">ms</span></div>
        </div>
        <div class="stat-box">
          <div class="stat-title">⌫ Backspaces</div>
          <div class="stat-val" id="statError">0%</div>
        </div>
      </div>

      <div class="result-banner" id="resultBanner">
        <div>
          <div style="font-size: 11px; color: var(--text-secondary); font-weight: 600; text-transform: uppercase;">Passive Behavioral Classification (Typing)</div>
          <div class="top-emotion" id="topEmotionLabel">Waiting for keystrokes...</div>
        </div>
        <button class="btn-reset" onclick="resetMetrics()">Clear Text</button>
      </div>

      <div class="diagnosis" id="diagnosisBox">
        Type at least 15-20 characters so the model can evaluate your motor cadence, key hold latencies, and flight rhythm.
      </div>

      <div class="prob-container" id="probContainer">
        <!-- Probability bars rendered dynamically -->
      </div>
    </div>

    <script>
      // Dynamic Theme Detection (Streamlit Dark / Light Mode)
      function detectTheme() {
        let isDark = false;
        try {
          const parentDoc = window.parent.document;
          const bodyBg = window.getComputedStyle(parentDoc.body).backgroundColor;
          const match = bodyBg.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
          if (match) {
            const r = parseInt(match[1]), g = parseInt(match[2]), b = parseInt(match[3]);
            const brightness = (r * 299 + g * 587 + b * 114) / 1000;
            isDark = brightness < 128;
          } else {
            isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
          }
        } catch (e) {
          isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        }

        if (isDark) {
          document.body.classList.add('dark-theme');
          document.body.classList.remove('light-theme');
        } else {
          document.body.classList.add('light-theme');
          document.body.classList.remove('dark-theme');
        }
      }

      detectTheme();
      setInterval(detectTheme, 1000);
      window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', detectTheme);

      const keyPressTimes = {};
      const holdTimes = [];
      const flightTimes = [];
      let lastKeyUpTime = 0;
      let totalKeystrokes = 0;
      let backspaceCount = 0;
      let startTime = 0;

      const EMOTIONS = [
        { name: "Neutral", color: "#a0aec0" },
        { name: "Calm", color: "#38a169" },
        { name: "Happy", color: "#3182ce" },
        { name: "Sad", color: "#9f7aea" },
        { name: "Angry", color: "#e53e3e" }
      ];

      const textarea = document.getElementById("typingArea");

      function setPrompt(text) {
        resetMetrics();
        textarea.value = text;
        textarea.focus();
        // Trigger simulated typing evaluation
        for (let i = 0; i < text.length; i++) {
          totalKeystrokes++;
        }
      }

      function resetMetrics() {
        textarea.value = "";
        holdTimes.length = 0;
        flightTimes.length = 0;
        lastKeyUpTime = 0;
        totalKeystrokes = 0;
        backspaceCount = 0;
        startTime = 0;
        for (let k in keyPressTimes) delete keyPressTimes[k];

        document.getElementById("statWpm").innerHTML = "0 <span style='font-size:11px; font-weight:normal;'>WPM</span>";
        document.getElementById("statHold").innerHTML = "0 <span style='font-size:11px; font-weight:normal;'>ms</span>";
        document.getElementById("statFlight").innerHTML = "0 <span style='font-size:11px; font-weight:normal;'>ms</span>";
        document.getElementById("statJitter").innerHTML = "0 <span style='font-size:11px; font-weight:normal;'>ms</span>";
        document.getElementById("statError").innerText = "0%";
        document.getElementById("topEmotionLabel").innerText = "Waiting for keystrokes...";
        document.getElementById("diagnosisBox").innerText = "Type at least 15-20 characters so the model can evaluate your motor cadence.";
        renderBars([0.2, 0.2, 0.2, 0.2, 0.2]);

        try {
          const parentUrl = new URL(window.parent.location.href);
          parentUrl.searchParams.delete("typing_emotion");
          parentUrl.searchParams.delete("typing_probs");
          window.parent.history.replaceState({}, "", parentUrl.toString());
          window.parent.dispatchEvent(new PopStateEvent('popstate'));
        } catch (e) {}
      }

      textarea.addEventListener("keydown", (e) => {
        const now = performance.now();
        if (!startTime) startTime = now;
        totalKeystrokes++;

        if (e.key === "Backspace" || e.key === "Delete") {
          backspaceCount++;
        }

        if (lastKeyUpTime > 0) {
          const ft = now - lastKeyUpTime;
          // Capture intra-burst flight latency between 20ms and 700ms
          // Long pauses (>700ms) are pauses to think, NOT motor rhythm jitter
          if (ft > 20 && ft <= 700) {
            flightTimes.push(ft);
          }
        }

        if (!keyPressTimes[e.code]) {
          keyPressTimes[e.code] = now;
        }
      });

      textarea.addEventListener("keyup", (e) => {
        const now = performance.now();
        lastKeyUpTime = now;

        if (keyPressTimes[e.code]) {
          const ht = now - keyPressTimes[e.code];
          if (ht > 15 && ht < 1000) {
            holdTimes.push(ht);
          }
          delete keyPressTimes[e.code];
        }

        updateAnalysis();
      });

      function updateAnalysis() {
        if (holdTimes.length < 8) {
          document.getElementById("topEmotionLabel").innerText = "Analyzing typing cadence...";
          document.getElementById("diagnosisBox").innerText = "Type at least 15-20 characters to capture your natural motor cadence and rhythm.";
          return;
        }

        const avgHold = holdTimes.reduce((a, b) => a + b, 0) / holdTimes.length;
        const avgFlight = flightTimes.length > 0 ? (flightTimes.reduce((a, b) => a + b, 0) / flightTimes.length) : 140;
        
        let jitter = 35;
        if (flightTimes.length > 2) {
          const variance = flightTimes.reduce((acc, val) => acc + Math.pow(val - avgFlight, 2), 0) / flightTimes.length;
          jitter = Math.sqrt(variance);
        }
        // Clamp jitter to realistic human biometric boundaries
        jitter = Math.min(Math.max(jitter, 15), 110);

        const elapsedMinutes = Math.max((performance.now() - startTime) / 60000, 0.05);
        const wpm = Math.min(Math.max(Math.round((textarea.value.length / 5) / elapsedMinutes), 5), 130);
        const errorRate = Math.min((backspaceCount / Math.max(totalKeystrokes, 1)) * 100, 45);

        document.getElementById("statWpm").innerHTML = Math.round(wpm) + " <span style='font-size:11px; font-weight:normal;'>WPM</span>";
        document.getElementById("statHold").innerHTML = Math.round(avgHold) + " <span style='font-size:11px; font-weight:normal;'>ms</span>";
        document.getElementById("statFlight").innerHTML = Math.round(avgFlight) + " <span style='font-size:11px; font-weight:normal;'>ms</span>";
        document.getElementById("statJitter").innerHTML = Math.round(jitter) + " <span style='font-size:11px; font-weight:normal;'>ms</span>";
        document.getElementById("statError").innerText = errorRate.toFixed(1) + "%";

        // Calibrated Bounded Z-Score Model
        const z_wpm_h = Math.max(-1.5, Math.min((wpm - 42) / 18.0, 2.0));
        const z_hold_h = Math.max(-1.5, Math.min((95 - avgHold) / 25.0, 2.0));
        const z_err_h = Math.max(-1.5, Math.min((4 - errorRate) / 5.0, 1.0));
        const s_happy = 0.5 * z_wpm_h + 0.6 * z_hold_h + 0.4 * z_err_h;

        const z_jit_c = Math.max(-1.5, Math.min((42 - jitter) / 18.0, 2.0));
        const z_err_c = Math.max(-1.5, Math.min((4 - errorRate) / 4.0, 1.5));
        const z_pace_c = Math.max(-1.0, Math.min(1.4 - Math.abs(wpm - 40) / 15.0, 1.5));
        const s_calm = 0.7 * z_jit_c + 0.5 * z_err_c + 0.4 * z_pace_c;

        const z_wpm_s = Math.max(-1.5, Math.min((35 - wpm) / 12.0, 2.0));
        const z_hold_s = Math.max(-1.5, Math.min((avgHold - 110) / 25.0, 2.0));
        const s_sad = 0.7 * z_wpm_s + 0.7 * z_hold_s;

        // Angry: requires BOTH elevated error/correction rate (>12%) AND erratic rhythm jitter (>55ms)
        const z_err_a = Math.max(-1.5, Math.min((errorRate - 12) / 8.0, 2.5));
        const z_jit_a = Math.max(-1.5, Math.min((jitter - 55) / 22.0, 2.0));
        const hold_penalty = avgHold > 130 ? (avgHold - 130) / 25.0 : 0;
        const s_angry = (0.75 * z_err_a + 0.65 * z_jit_a) - hold_penalty;

        const max_dev = Math.max(s_happy, s_calm, s_sad, s_angry);
        const s_neutral = 0.3 - (0.45 * max_dev);

        const rawScores = [s_neutral, s_calm, s_happy, s_sad, s_angry];
        const maxScore = Math.max(...rawScores);
        const expScores = rawScores.map(s => Math.exp(s - maxScore));
        const sumExp = expScores.reduce((a, b) => a + b, 0);
        const probs = expScores.map(s => s / sumExp);

        let topIdx = 0;
        let maxP = 0;
        for (let i = 0; i < probs.length; i++) {
          if (probs[i] > maxP) {
            maxP = probs[i];
            topIdx = i;
          }
        }

        const topEmotion = EMOTIONS[topIdx];
        const banner = document.getElementById("resultBanner");
        banner.style.borderLeft = "5px solid " + topEmotion.color;
        document.getElementById("topEmotionLabel").innerHTML = `<span style="color:${topEmotion.color}">● ${topEmotion.name}</span> <span style="font-size:14px; color:var(--text-secondary); font-weight:normal;">(${(maxP*100).toFixed(1)}%)</span>`;

        let diag = "";
        if (topEmotion.name === "Angry") {
          diag = `Frequent corrections (${errorRate.toFixed(1)}%) combined with elevated rhythm jitter (${Math.round(jitter)}ms) indicates motor agitation and cognitive friction.`;
        } else if (topEmotion.name === "Calm") {
          diag = `Smooth, rhythmic motor cadence with low jitter (${Math.round(jitter)}ms) and minimal error corrections reflects a relaxed, focused state.`;
        } else if (topEmotion.name === "Happy") {
          diag = `Brisk typing speed (${Math.round(wpm)} WPM) with swift key hold latencies (${Math.round(avgHold)}ms) reflects energetic positive valence.`;
        } else if (topEmotion.name === "Sad") {
          diag = `Slow typing velocity (${Math.round(wpm)} WPM) with prolonged key hold latencies (${Math.round(avgHold)}ms) reflects psychomotor fatigue.`;
        } else {
          diag = `Typing velocity and rhythm match baseline equilibrium metrics.`;
        }
        document.getElementById("diagnosisBox").innerText = diag;
        renderBars(probs);

        try {
          const parentUrl = new URL(window.parent.location.href);
          parentUrl.searchParams.set("typing_emotion", topEmotion.name);
          parentUrl.searchParams.set("typing_probs", JSON.stringify(probs));
          window.parent.history.replaceState({}, "", parentUrl.toString());
          window.parent.dispatchEvent(new PopStateEvent('popstate'));
        } catch (e) {}
      }

      function renderBars(probs) {
        const container = document.getElementById("probContainer");
        let html = "";
        EMOTIONS.forEach((em, i) => {
          const pct = ((probs[i] || 0) * 100).toFixed(1);
          html += `
            <div class="prob-row">
              <span class="prob-label" style="color:${em.color}">${em.name}</span>
              <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width:${pct}%; background:${em.color};"></div>
              </div>
              <span class="prob-val">${pct}%</span>
            </div>
          `;
        });
        container.innerHTML = html;
      }

      renderBars([0.2, 0.2, 0.2, 0.2, 0.2]);
    </script>
    </body>
    </html>
    """

components.html(typing_html_component, height=590, scrolling=False)

st.subheader("8. Voice Check-in: How do you feel right now? (Reply by Voice)")
st.caption(
    "🎙️ **Acoustic Voice Sensing:** Speak or upload a voice recording describing your feelings. "
    "Your voice will be analyzed using 80 MFCC features through the trained Random Forest classifier."
)

audio_mode = st.radio(
    "Choose voice input method:",
    ["Record voice", "Upload audio file"],
    horizontal=True,
    key="audio_mode_q8"
)

if audio_mode == "Record voice":
    audio_file = st.audio_input("Record your voice", key="q8_record")
else:
    audio_file = st.file_uploader(
        "Upload audio file (.wav, .mp3, .ogg, .flac, .m4a)",
        type=["wav", "mp3", "ogg", "flac", "m4a"],
        key="q8_upload"
    )

if audio_file is not None:
    st.audio(audio_file)

st.markdown("---")

if st.button("🚀 Analyze Multimodal Assessment", type="primary"):
    st.header("📊 Multimodal Assessment Results")

    p_survey = None
    p_voice = None
    p_typing = None

    # ---------------- 1. Self-Reported Baseline ----------------
    st.subheader("1. 📝 Subjective Self-Report Baseline")
    scores = {
        "Happy": happy,
        "Calm": calm,
        "Sad": sad,
        "Angry": angry,
        "Neutral": neutral
    }

    has_survey_responses = (sum(scores.values()) > 0) or (reported_emotion is not None and reported_emotion in CLASSES)

    if has_survey_responses:
        # Probability distribution for Subjective Self-Report Baseline
        survey_raw = np.array([float(scores.get(c, 0)) for c in CLASSES])
        if reported_emotion and reported_emotion in CLASSES:
            survey_raw[CLASSES.index(reported_emotion)] += 1.5
        if survey_raw.sum() > 0:
            p_survey = survey_raw / survey_raw.sum()

        highest_score = max(scores.values())
        if highest_score > 0:
            top_emotions = [
                emotion for emotion, score in scores.items()
                if score == highest_score
            ]
            if len(top_emotions) > 1:
                st.info("Self-reported responses indicate a mixed emotional state: " + ", ".join(top_emotions))
            else:
                st.success(f"Self-reported rule-based estimate: **{top_emotions[0]}**")
        elif reported_emotion:
            st.success(f"Self-reported categorical label: **{reported_emotion}**")

        for emotion, score in scores.items():
            st.write(f"**{emotion}:** {score}/4")
            st.progress(score / 4)
        if reported_emotion:
            st.write(f"**Categorical self-label selected:** {reported_emotion}")
    else:
        p_survey = None
        st.info("ℹ️ No questionnaire responses provided in Questions 1–6. Subjective self-report baseline not included.")

    st.markdown("---")

    # ---------------- 2. Voice Acoustic Model ----------------
    st.subheader("2. 🎙️ Acoustic Voice Emotion (Trained MFCC Model)")
    if audio_file is not None:
        try:
            audio, sr = sf.read(io.BytesIO(audio_file.getvalue()))
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

            if len(audio) == 0:
                st.error("The recording is empty.")
            else:
                mfcc = librosa.feature.mfcc(
                    y=audio.astype(np.float32),
                    sr=sr,
                    n_mfcc=40
                )
                features = np.concatenate([
                    np.mean(mfcc, axis=1),
                    np.std(mfcc, axis=1)
                ]).reshape(1, -1)

                prediction = int(model.predict(features)[0])
                pred_label = CLASSES[prediction]
                st.success(f"Predicted Voice Emotion: **{pred_label}**")

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(features)[0]
                    p_voice = probabilities
                    st.write("Acoustic Model Confidence Scores:")
                    for label, score in zip(CLASSES, probabilities):
                        st.write(f"{label}: {score:.1%}")
                        st.progress(float(score))
        except Exception as e:
            st.error(f"Voice emotion analysis failed: {e}")
    else:
        st.info("ℹ️ No voice audio recorded in Question 8. Record or upload voice audio to include acoustic emotion detection.")

    st.markdown("---")

    # ---------------- 3. Behavioral Typing Dynamics ----------------
    st.subheader("3. ⌨️ Behavioral Sensing (Typing Dynamics)")
    typing_emotion = st.query_params.get("typing_emotion", None)
    typing_probs_raw = st.query_params.get("typing_probs", None)

    if isinstance(typing_emotion, list):
        typing_emotion = typing_emotion[0] if typing_emotion else None
    if isinstance(typing_probs_raw, list):
        typing_probs_raw = typing_probs_raw[0] if typing_probs_raw else None

    if typing_emotion and typing_probs_raw:
        try:
            t_probs = json.loads(typing_probs_raw)
            p_typing = np.array(t_probs, dtype=float)
            top_prob_val = max(t_probs)
            st.success(f"Predicted Behavioral Emotion: **{typing_emotion}** ({top_prob_val:.1%})")

            st.write("Behavioral Model Confidence Scores:")
            for label, prob in zip(CLASSES, t_probs):
                st.write(f"{label}: {float(prob):.1%}")
                st.progress(float(prob))
        except Exception:
            st.success(f"Predicted Behavioral Emotion: **{typing_emotion}**")
    else:
        st.info("ℹ️ No typing behavior detected in Question 7. Type a sentence in Question 7 above to include behavioral typing analysis.")

        st.markdown("---")

            # ---------------- 4. Multimodal Affective State Fusion ----------------
    st.subheader("4. 🌐 Fused Multimodal Affective State (Final Output)")
    active_modalities = []
    if p_voice is not None:
        active_modalities.append(("Voice Acoustic Sensing", 0.45, np.array(p_voice, dtype=float)))
    if p_typing is not None:
        active_modalities.append(("Typing Keystroke Dynamics", 0.35, np.array(p_typing, dtype=float)))
    if p_survey is not None:
        active_modalities.append(("Subjective Self-Report", 0.20, np.array(p_survey, dtype=float)))
    if active_modalities:
        total_weight = sum(w for _, w, _ in active_modalities)
        fused_probs = np.zeros(len(CLASSES))
        contributing_labels = []
        for name, weight, probs in active_modalities:
            normalized_w = weight / total_weight
            fused_probs += normalized_w * probs
            contributing_labels.append(f"**{name}** ({normalized_w:.0%})")
        fused_pred_idx = int(np.argmax(fused_probs))
        fused_emotion = CLASSES[fused_pred_idx]
        fused_confidence = fused_probs[fused_pred_idx]
        st.success(f"🏆 Final Fused Affective State: **{fused_emotion}** ({fused_confidence:.1%})")
        st.caption(
            "🔗 **Decision-Level Late Fusion:** Dynamically synthesized from active modalities: " +
            " + ".join(contributing_labels)
        )
        st.write("Fused Multimodal Model Confidence Scores:")
        for label, prob in zip(CLASSES, fused_probs):
            st.write(f"{label}: {float(prob):.1%}")
            st.progress(float(prob))
    else:
        st.warning("⚠️ No modalities available to perform multimodal fusion.")

st.markdown("---")

with st.expander("🧪 Biometric Parameter Calibrator & Model Testing (Python Inference)"):
    st.write(
        "Use these controls to inspect how individual keystroke biometric parameters shift the "
        "singular behavioral prediction model across **Neutral, Calm, Happy, Sad, and Angry**."
    )

    col_left, col_right = st.columns(2)

    with col_left:
        sim_wpm = st.slider("Typing Speed (WPM)", min_value=10, max_value=120, value=42, step=1, key="sim_wpm")
        sim_hold = st.slider("Key Hold Time (ms)", min_value=40, max_value=250, value=110, step=5, key="sim_hold")
        sim_flight = st.slider("Flight Time / Inter-Key Interval (ms)", min_value=40, max_value=350, value=150, step=5, key="sim_flight")

    with col_right:
        sim_jitter = st.slider("Rhythm Jitter / StdDev of Flight (ms)", min_value=10, max_value=180, value=45, step=5, key="sim_jitter")
        sim_error = st.slider("Backspace / Error Rate (%)", min_value=0.0, max_value=50.0, value=5.0, step=0.5, key="sim_error")

    if st.button("Run Python Behavioral Prediction", type="primary", key="btn_sim"):
        sim_probs = predict_typing_emotion(sim_wpm, sim_hold, sim_flight, sim_jitter, sim_error)
        top_class_idx = int(np.argmax(sim_probs))
        top_class = CLASSES[top_class_idx]

        st.success(f"Predicted Behavioral Affect: **{top_class}** ({sim_probs[top_class_idx]:.1%})")

        st.subheader("Model Scores across Affective Classes")
        for label, prob in zip(CLASSES, sim_probs):
            st.write(f"**{label}:** {prob:.1%}")
            st.progress(float(prob))

st.caption(
    "Research Prototype: Multimodal Affective State Detection (Singular Models, 5 Classes). "
    "Fulfills the Voice, Typing, and Questionnaire specifications of the system architecture."
)


