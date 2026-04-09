"""
LaptopLens - Smart Laptop Price Advisor
========================================
Fixed version: all categorical values match exactly what the pipeline's
OneHotEncoder was trained on (extracted from notebook outputs).

Root cause of the original ValueError:
  Gpu_convertor() in the notebook returns LOWERCASE strings:
    'intel', 'nvidia', 'amd', 'others'
  The old app was sending 'Intel', 'Nvidia' etc. → OHE had never seen those.

Fix: GPU_DISPLAY dict maps user-friendly labels → exact lowercase pipeline values.
     Same pattern applied to Os Brand for safety.

Pipeline column order (indices [0,1,6,10,11] passed to OHE):
  0  Company | 1 TypeName | 6 Cpu brand | 10 Gpu Brand | 11 Os Brand
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page config — must be the FIRST Streamlit call ───────────────────────────
st.set_page_config(
    page_title="LaptopLens – Smart Price Advisor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
LOG_MAE               = 0.147   # best ensemble MAE from notebook (log-scale)
OVERPRICED_THRESHOLD  =  0.12   # asking > fair + 12%  → Overpriced
UNDERPRICED_THRESHOLD = -0.10   # asking < fair - 10%  → Great Deal

# Exact values the OHE was fitted on — verified from notebook Cell 68/70 outputs
COMPANIES = sorted([
    "Apple", "Asus", "Chuwi", "Dell", "Fujitsu", "Google",
    "HP", "Huawei", "LG", "Lenovo", "MSI", "Mediacom",
    "Microsoft", "Razer", "Samsung", "Toshiba", "Vero", "Xiaomi",
])

TYPE_NAMES = [
    "2 in 1 Convertible", "Gaming", "Netbook",
    "Notebook", "Ultrabook", "Workstation",
]

CPU_BRANDS = [
    "Intel Core i3", "Intel Core i5", "Intel Core i7",
    "AMD", "other cpu",
]

# ⚠️  CRITICAL: Gpu_convertor() returns lowercase — must match exactly
GPU_DISPLAY = {
    "Intel (Integrated)": "intel",
    "Nvidia (Dedicated)": "nvidia",
    "AMD (Dedicated)":    "amd",
    "Others":             "others",
}

# Os Brand values from os_convertor() — note 'MacOs' (capital O, lowercase s)
OS_DISPLAY = {
    "Windows":                  "Windows",
    "macOS":                    "MacOs",
    "Linux / Chrome / Android": "others/linux/Android",
}

RAM_OPTIONS = [2, 4, 6, 8, 12, 16, 24, 32, 64]
HDD_OPTIONS = [0, 500, 1000, 2000]
SSD_OPTIONS = [0, 8, 128, 256, 512, 1000]


# ── Model loader ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ML model…")
def load_model():
    path = os.path.join(os.path.dirname(__file__), "pipe.joblib")
    return joblib.load(path) if os.path.exists(path) else None


# ── Prediction ────────────────────────────────────────────────────────────────
def predict_price(pipe, features: dict) -> dict:
    """
    Single-row prediction through the saved pipeline.
    Columns must arrive in this exact order (matching x.head() output):
      Company, TypeName, Ram, Weight, Touchscreen, Ips,
      Cpu brand, HDD, SSD, ppi, Gpu Brand, Os Brand
    Target was log(Price), so we exp() the result back to INR.
    """
    col_order = [
        "Company", "TypeName", "Ram", "Weight", "Touchscreen", "Ips",
        "Cpu brand", "HDD", "SSD", "ppi", "Gpu Brand", "Os Brand",
    ]
    df = pd.DataFrame([{c: features[c] for c in col_order}])

    log_pred = pipe.predict(df)[0]
    return {
        "predicted_inr": int(np.exp(log_pred)),
        "lower_inr":     int(np.exp(log_pred - LOG_MAE)),
        "upper_inr":     int(np.exp(log_pred + LOG_MAE)),
    }


# ── Deal verdict ──────────────────────────────────────────────────────────────
def deal_verdict(asking: int, predicted: int) -> dict | None:
    if asking <= 0:
        return None
    delta = (asking - predicted) / predicted
    if delta > OVERPRICED_THRESHOLD:
        return dict(label="Overpriced", colour="#ff4b4b", emoji="🔴", delta=delta,
                    message=f"Asking price is **{abs(delta)*100:.1f}% above** fair market value. Negotiate hard or look for alternatives.")
    elif delta < UNDERPRICED_THRESHOLD:
        return dict(label="Great Deal", colour="#21c354", emoji="🟢", delta=delta,
                    message=f"Asking price is **{abs(delta)*100:.1f}% below** fair market value. This looks like a steal — act fast!")
    else:
        return dict(label="Fair Price", colour="#f4a261", emoji="🟡", delta=delta,
                    message=f"Asking price is within ±{abs(delta)*100:.1f}% of fair value. This is a reasonable deal.")


# ── Insight engine ────────────────────────────────────────────────────────────
def spec_insights(features: dict) -> list[str]:
    tips = []
    if features["Ram"] <= 4:
        tips.append("⚠️ **4 GB RAM** is limiting for modern multitasking. An 8 GB variant typically costs ₹5,000–₹8,000 more.")
    elif features["Ram"] >= 16:
        tips.append("✅ **16 GB+ RAM** — future-proof for development and heavy workloads.")

    if features["SSD"] == 0:
        tips.append("⚠️ **No SSD.** HDD-only laptops feel noticeably slower. SSD adds ₹3,000–₹6,000 in resale value.")
    elif features["SSD"] >= 512:
        tips.append("✅ **512 GB+ SSD** — ample storage with strong resale value.")

    if features["Touchscreen"] == 1:
        tips.append("📌 **Touchscreen** commands a ₹4,000–₹7,000 premium over non-touch equivalents.")
    if features["Ips"] == 1:
        tips.append("✅ **IPS panel** — better colour accuracy and wider viewing angles.")

    gpu = features["Gpu Brand"]
    if gpu == "nvidia":
        tips.append("🎮 **Nvidia GPU** adds significant value for gaming, ML, and video editing.")
    elif gpu == "intel":
        tips.append("📌 **Integrated Intel GPU** — fine for everyday use; not suitable for gaming.")

    if features["Os Brand"] == "MacOs":
        tips.append("🍎 **macOS** laptops carry a strong brand premium; resale value holds well.")

    if features["Weight"] < 1.5:
        tips.append(f"🪶 **Ultra-light at {features['Weight']} kg** — premium portability tier.")
    elif features["Weight"] > 2.5:
        tips.append(f"🏋️ **Heavy at {features['Weight']} kg** — typical of gaming/workstation builds.")

    return tips


def upgrade_suggestions(features: dict) -> list[str]:
    s = []
    if features["Ram"] < 16:
        s.append(f"💡 Upgrading RAM **{features['Ram']} GB → 16 GB** typically adds ₹4,000–₹10,000 to fair value.")
    if 0 < features["SSD"] < 512:
        s.append("💡 Doubling SSD to **512 GB** usually raises fair price by ₹3,000–₹6,000.")
    if features["SSD"] == 0:
        s.append("💡 Adding any **SSD** (even 128 GB) dramatically improves performance and resale value.")
    if features["Touchscreen"] == 0 and features["TypeName"] in ("2 in 1 Convertible", "Ultrabook"):
        s.append("💡 A **touchscreen** variant of this form factor typically costs ₹5,000–₹8,000 more.")
    if features["Gpu Brand"] == "intel" and features["TypeName"] == "Gaming":
        s.append("⚠️ A **Gaming** laptop with integrated Intel GPU is unusual — verify the spec sheet before buying.")
    return s


# ── CSS ───────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
        .stApp { background-color: #0f1117; }
        .section-title {
            font-size: 0.78rem; font-weight: 700; color: #a78bfa;
            letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.4rem;
        }
        .pred-card {
            background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
            border: 1px solid #4f46e5; border-radius: 14px;
            padding: 1.6rem 2rem; text-align: center;
        }
        .pred-amount { font-size: 2.8rem; font-weight: 800; color: #a78bfa; }
        .pred-range  { font-size: 0.92rem; color: #94a3b8; margin-top: 0.3rem; }
        .verdict-card { border-radius: 10px; padding: 1rem 1.5rem; margin-top: 0.8rem; }
        .verdict-title { font-size: 1.5rem; font-weight: 800; }
        .tip-pill {
            background: #1e293b; border-left: 3px solid #4f46e5; border-radius: 6px;
            padding: 0.55rem 0.9rem; margin-bottom: 0.45rem; font-size: 0.91rem;
        }
        .spec-row {
            display: flex; justify-content: space-between;
            padding: 0.3rem 0; border-bottom: 1px solid #1e293b; font-size: 0.9rem;
        }
        .spec-key { color: #94a3b8; }
        .spec-val { color: #e2e8f0; font-weight: 500; }
        .footer {
            text-align: center; color: #475569; font-size: 0.78rem;
            margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #1e293b;
        }
        #MainMenu { visibility: hidden; } footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown("## 🔧 Laptop Specifications")
        st.caption("Enter the full spec sheet of the laptop you want to evaluate.")

        st.markdown('<p class="section-title">Brand & Category</p>', unsafe_allow_html=True)
        company   = st.selectbox("Manufacturer", COMPANIES, index=COMPANIES.index("Dell"))
        type_name = st.selectbox("Laptop Type",  TYPE_NAMES, index=TYPE_NAMES.index("Notebook"))

        st.markdown('<p class="section-title">Display</p>', unsafe_allow_html=True)
        inches     = st.slider("Screen Size (inches)", 10.0, 18.0, 15.6, step=0.1)
        resolution = st.selectbox("Screen Resolution", [
            "1366x768 (HD)", "1920x1080 (Full HD)", "2560x1440 (QHD)",
            "3840x2160 (4K)", "2880x1800 (Retina)",
        ], index=1)
        touchscreen = st.checkbox("Touchscreen", value=False)
        ips         = st.checkbox("IPS Panel",   value=True)

        res_map = {
            "1366x768 (HD)":       (1366, 768),
            "1920x1080 (Full HD)": (1920, 1080),
            "2560x1440 (QHD)":     (2560, 1440),
            "3840x2160 (4K)":      (3840, 2160),
            "2880x1800 (Retina)":  (2880, 1800),
        }
        width, height = res_map[resolution]
        ppi = round(((width**2 + height**2) ** 0.5) / inches, 4)

        st.markdown('<p class="section-title">Processor</p>', unsafe_allow_html=True)
        cpu_brand = st.selectbox("CPU Brand / Generation", CPU_BRANDS, index=1)

        st.markdown('<p class="section-title">Memory & Storage</p>', unsafe_allow_html=True)
        ram = st.select_slider("RAM (GB)",  options=RAM_OPTIONS, value=8)
        hdd = st.select_slider("HDD (GB)",  options=HDD_OPTIONS, value=0)
        ssd = st.select_slider("SSD (GB)",  options=SSD_OPTIONS, value=256)

        st.markdown('<p class="section-title">Graphics & OS</p>', unsafe_allow_html=True)
        gpu_label = st.selectbox("GPU Brand",        list(GPU_DISPLAY.keys()), index=0)
        os_label  = st.selectbox("Operating System", list(OS_DISPLAY.keys()),  index=0)

        st.markdown('<p class="section-title">Build</p>', unsafe_allow_html=True)
        weight = st.slider("Weight (kg)", 0.8, 5.0, 1.8, step=0.1)

        st.divider()
        st.markdown("### 💰 Asking Price (Optional)")
        st.caption("Enter the listed price to see a deal verdict.")
        asking_price = st.number_input(
            "Asking Price (₹)", min_value=0, max_value=500000, value=0, step=1000,
        )

    return {
        "features": {
            "Company":     company,
            "TypeName":    type_name,
            "Ram":         ram,
            "Weight":      float(weight),
            "Touchscreen": int(touchscreen),
            "Ips":         int(ips),
            "Cpu brand":   cpu_brand,
            "HDD":         hdd,
            "SSD":         ssd,
            "ppi":         ppi,
            "Gpu Brand":   GPU_DISPLAY[gpu_label],   # ← lowercase: 'intel' etc.
            "Os Brand":    OS_DISPLAY[os_label],     # ← 'MacOs' etc.
        },
        "meta": {
            "inches": inches, "resolution": resolution,
            "ppi": ppi, "asking_price": asking_price,
            "gpu_label": gpu_label, "os_label": os_label,
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def render_main(pipe, user_input: dict):
    features     = user_input["features"]
    meta         = user_input["meta"]
    asking_price = meta["asking_price"]

    st.markdown(
        "<h1 style='color:#a78bfa; margin-bottom:0;'>💻 LaptopLens</h1>"
        "<p style='color:#94a3b8; font-size:1.05rem; margin-top:0;'>"
        "AI-powered fair price intelligence for smarter laptop buying</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # Run prediction
    try:
        result = predict_price(pipe, features)
    except Exception as e:
        st.error(f"**Prediction error:** {e}")
        st.stop()

    col_pred, col_meta = st.columns([1, 1], gap="large")

    with col_pred:
        st.markdown(f"""
        <div class="pred-card">
            <div style="color:#94a3b8;font-size:0.85rem;letter-spacing:0.06em;margin-bottom:0.4rem;">
                ESTIMATED FAIR MARKET PRICE
            </div>
            <div class="pred-amount">₹{result['predicted_inr']:,}</div>
            <div class="pred-range">
                Confidence band &nbsp;·&nbsp; ₹{result['lower_inr']:,} – ₹{result['upper_inr']:,}
            </div>
            <div style="color:#475569;font-size:0.78rem;margin-top:0.6rem;">
                ±{LOG_MAE*100:.0f}% model MAE propagated from log-scale
            </div>
        </div>
        """, unsafe_allow_html=True)

        if asking_price > 0:
            v = deal_verdict(asking_price, result["predicted_inr"])
            if v:
                d = f"{'+'if v['delta']>0 else ''}{v['delta']*100:.1f}%"
                st.markdown(f"""
                <div class="verdict-card"
                     style="background:{v['colour']}18;border:1px solid {v['colour']};">
                    <div class="verdict-title" style="color:{v['colour']};">
                        {v['emoji']} {v['label']}
                        <span style="font-size:1rem;font-weight:400;">({d})</span>
                    </div>
                    <div style="color:#cbd5e1;font-size:0.93rem;margin-top:0.4rem;">
                        {v['message']}
                    </div>
                    <div style="color:#64748b;font-size:0.82rem;margin-top:0.6rem;">
                        Listed ₹{asking_price:,} &nbsp;·&nbsp;
                        Fair ₹{result['predicted_inr']:,} &nbsp;·&nbsp;
                        Diff ₹{asking_price - result['predicted_inr']:+,}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Enter an **Asking Price** in the sidebar to get a deal verdict.", icon="💡")

    with col_meta:
        st.markdown('<p class="section-title">📋 Spec Summary</p>', unsafe_allow_html=True)
        rows = [
            ("Manufacturer",  features["Company"]),
            ("Type",          features["TypeName"]),
            ("Display",       f"{meta['inches']}\"  {meta['resolution']}"),
            ("Panel / Touch", f"{'IPS' if features['Ips'] else 'non-IPS'} / {'Touch' if features['Touchscreen'] else 'No Touch'}"),
            ("PPI",           f"{meta['ppi']:.1f}"),
            ("CPU",           features["Cpu brand"]),
            ("RAM",           f"{features['Ram']} GB"),
            ("SSD",           f"{features['SSD']} GB"),
            ("HDD",           f"{features['HDD']} GB"),
            ("GPU",           meta["gpu_label"]),
            ("OS",            meta["os_label"]),
            ("Weight",        f"{features['Weight']} kg"),
        ]
        html = "".join(
            f'<div class="spec-row"><span class="spec-key">{k}</span>'
            f'<span class="spec-val">{v}</span></div>'
            for k, v in rows
        )
        st.markdown(html, unsafe_allow_html=True)

    st.divider()

    col_ins, col_upg = st.columns(2, gap="large")
    with col_ins:
        st.markdown('<p class="section-title">💡 Spec Insights</p>', unsafe_allow_html=True)
        for tip in spec_insights(features) or ["✅ All specs look well-balanced."]:
            st.markdown(f'<div class="tip-pill">{tip}</div>', unsafe_allow_html=True)

    with col_upg:
        st.markdown('<p class="section-title">🚀 Upgrade Suggestions</p>', unsafe_allow_html=True)
        for sug in upgrade_suggestions(features) or ["✅ Configuration is already strong."]:
            st.markdown(f'<div class="tip-pill">{sug}</div>', unsafe_allow_html=True)

    st.divider()

    if asking_price > 0:
        st.markdown('<p class="section-title">📊 Price Comparison</p>', unsafe_allow_html=True)
        bar_df = pd.DataFrame({
            "Category": ["Lower Bound", "Fair Price", "Upper Bound", "Asking Price"],
            "Price (₹)": [
                result["lower_inr"], result["predicted_inr"],
                result["upper_inr"], asking_price,
            ],
        }).set_index("Category")
        st.bar_chart(bar_df, color="#a78bfa", height=260)
        st.caption("Lower/Upper bounds = ±MAE confidence interval. Asking price above upper bound → negotiate.")
        st.divider()

    with st.expander("📤 Copy Spec Report"):
        lines = [
            "Laptop Spec Evaluation — LaptopLens", "=" * 42,
            f"Brand / Type  : {features['Company']} {features['TypeName']}",
            f"Display       : {meta['inches']}\"  {meta['resolution']}",
            f"Panel / Touch : {'IPS' if features['Ips'] else 'non-IPS'} / {'Touch' if features['Touchscreen'] else 'No Touch'}",
            f"CPU           : {features['Cpu brand']}",
            f"RAM           : {features['Ram']} GB",
            f"Storage       : SSD {features['SSD']} GB | HDD {features['HDD']} GB",
            f"GPU           : {meta['gpu_label']}",
            f"OS            : {meta['os_label']}",
            f"Weight        : {features['Weight']} kg", "=" * 42,
            f"AI Fair Price : ₹{result['predicted_inr']:,}",
            f"Confidence    : ₹{result['lower_inr']:,} – ₹{result['upper_inr']:,}",
        ]
        if asking_price > 0:
            v = deal_verdict(asking_price, result["predicted_inr"])
            lines += [
                f"Asking Price  : ₹{asking_price:,}",
                f"Verdict       : {v['emoji']} {v['label']}  ({v['delta']*100:+.1f}%)",
            ]
        lines.append("Powered by LaptopLens ML Engine")
        st.code("\n".join(lines), language="")

    st.markdown(
        "<div class='footer'>LaptopLens &nbsp;|&nbsp; "
        "Stacking Regressor (RF + XGBoost + ExtraTrees) &nbsp;|&nbsp; "
        "Trained on 1,300 listings &nbsp;|&nbsp; Indicative purposes only.</div>",
        unsafe_allow_html=True,
    )


def main():
    inject_css()
    pipe = load_model()
    if pipe is None:
        st.error(
            "**`pipe.joblib` not found.**\n\n"
            "Generate it by running this in your notebook:\n"
            "```python\nimport joblib\njoblib.dump(pipe, 'pipe.joblib', compress=3)\n```\n"
            "Then place it in the same folder as `app.py`."
        )
        st.stop()
    render_main(pipe, render_sidebar())


if __name__ == "__main__":
    main()
