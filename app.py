import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import plotly.graph_objects as go
import pickle
import os
import datetime
import random

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Adib M&V Web Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
    /* Animated background for login card */
    @keyframes gradient {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    .login-bg {
        background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        padding: 2.5rem;
        border-radius: 25px;
        box-shadow: 0 0 40px rgba(0,255,255,0.2);
    }
    .glow {
        text-shadow: 0 0 15px #ffaa00, 0 0 25px #ffaa00;
    }
    /* Step indicator circles */
    .step-circle {
        width: 45px;
        height: 45px;
        border-radius: 50%;
        background-color: #4b5563;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        font-size: 1.2rem;
        transition: 0.3s;
    }
    .step-circle.active {
        background-color: #ff4b4b;
        box-shadow: 0 0 15px #ff4b4b;
    }
    .step-line {
        width: 80px;
        border-top: 3px solid #4b5563;
        align-self: center;
    }
    .step-line.active {
        border-top: 3px solid #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

# ---------- SESSION & MODEL LOADING ----------
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'role' not in st.session_state:
    st.session_state['role'] = None
if 'model_data' not in st.session_state or st.session_state['model_data'] is None:
    if os.path.exists("saved_baseline.pkl"):
        with open("saved_baseline.pkl", "rb") as f:
            st.session_state['model_data'] = pickle.load(f)
    else:
        st.session_state['model_data'] = None

def trigger_logout():
    st.session_state['logged_in'] = False
    st.session_state['role'] = None

# ---------- ENERGY FACTS ----------
energy_facts = [
    "💡 Switching to LED bulbs can save up to 75% of lighting energy.",
    "🌍 Buildings consume nearly 40% of global energy.",
    "⚡ A 1°C thermostat adjustment can save ~10% on heating/cooling.",
    "📊 Measurement & Verification (M&V) proves real savings.",
    "🏢 Commercial buildings with energy baselines reduce costs by 20% on average.",
    "🔋 A well‑tuned baseline model can detect even 5% energy drift."
]

# ===================== LOGIN SCREEN =====================
if not st.session_state['logged_in']:
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        st.markdown("<div class='login-bg'>", unsafe_allow_html=True)
        st.markdown("<h1 class='glow' style='text-align:center;'>⚡ ENERGY BASELINE PORTAL</h1>", unsafe_allow_html=True)
        st.write("")
        # Random fact
        st.info(f"**Did you know?** {random.choice(energy_facts)}")
        st.write("")
        username = st.text_input("👤 Username")
        password = st.text_input("🔒 Password", type="password")
        st.caption("Try: `adib` / `admin123`  or  `staff` / `user123`")
        if st.button("Login", type="primary", use_container_width=True):
            if username == "adib" and password == "admin123":
                st.session_state['logged_in'] = True
                st.session_state['role'] = "Admin"
                st.rerun()
            elif username == "staff" and password == "user123":
                st.session_state['logged_in'] = True
                st.session_state['role'] = "User"
                st.rerun()
            else:
                st.error("❌ Invalid credentials")
        st.markdown("</div>", unsafe_allow_html=True)

# ===================== AUTHENTICATED APP =====================
else:
    # Sidebar
    st.sidebar.title(f"👋 Welcome, {st.session_state['role']}")
    st.sidebar.button("🚪 Logout", on_click=trigger_logout)
    st.sidebar.markdown("---")
    st.sidebar.caption("Adib Affandi | Energy Baseline M&V")
    
    # Demo CSV download
    demo_csv = "Temperature,Production,Energy\n25,100,500\n26,110,520\n24,90,480\n27,105,540\n28,115,560"
    st.sidebar.download_button(
        label="📥 Download Demo Baseline CSV",
        data=demo_csv,
        file_name="demo_baseline.csv",
        mime="text/csv"
    )

    # Main title
    st.title("⚡ Energy Baseline M&V Tool")

    # ---------- STEP INDICATOR ----------
    st.markdown("## Workflow")
    model_ready = st.session_state['model_data'] is not None
    step1_class = "active"  # always active when logged in
    step2_class = "active" if model_ready else ""
    line_class = "active" if model_ready else ""

    st.markdown(f"""
    <div style="display:flex; gap:10px; justify-content:center; align-items:center; margin-bottom:25px;">
        <div class="step-circle {step1_class}">1</div>
        <span style="font-weight:600;">Setup Baseline</span>
        <div class="step-line {line_class}"></div>
        <div class="step-circle {step2_class}">2</div>
        <span style="font-weight:600;">Calculate Savings</span>
    </div>
    """, unsafe_allow_html=True)

    # ---------- METRIC CARDS (if model exists) ----------
    if model_ready:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 R² Score", f"{st.session_state['model_data']['r2']:.3f}")
        with col2:
            st.metric("📈 Variables", len(st.session_state['model_data']['vars']))
        with col3:
            if os.path.exists("saved_baseline.pkl"):
                mtime = os.path.getmtime("saved_baseline.pkl")
                last_train = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
                days_ago = (datetime.datetime.now() - datetime.datetime.fromtimestamp(mtime)).days
                st.metric("🕒 Last Trained", last_train, delta=f"{days_ago}d ago", delta_color="off")
            else:
                st.metric("🕒 Last Trained", "N/A")
        with col4:
            if os.path.exists("saved_baseline.pkl"):
                mtime = os.path.getmtime("saved_baseline.pkl")
                days = (datetime.datetime.now() - datetime.datetime.fromtimestamp(mtime)).days
                if days <= 30:
                    status = "🟢 Healthy"
                elif days <= 60:
                    status = "🟡 Stale"
                else:
                    status = "🔴 Outdated"
            else:
                status = "❓ No Model"
            st.metric("💚 Model Status", status)
        st.markdown("---")

    # ========== ADMIN: STEP 1 ==========
    if st.session_state['role'] == "Admin":
        st.header("📊 Step 1: Admin Baseline Setup")
        base_file = st.file_uploader("Upload Baseline CSV", type=['csv'], key="base")

        if base_file:
            df_baseline = pd.read_csv(base_file)
            st.write("Preview:", df_baseline.head(3))

            cols = list(df_baseline.columns)
            target_y = st.selectbox("Select Energy/Electricity Column (Y):", cols)
            selected_x_vars = st.multiselect("Select Independent Variables (X):", cols)

            if st.button("🔬 Run MLR Baseline Analysis", type="primary"):
                if target_y and selected_x_vars:
                    clean_df = df_baseline[selected_x_vars + [target_y]].dropna()
                    X = clean_df[selected_x_vars]
                    y = clean_df[target_y]

                    model = LinearRegression()
                    model.fit(X, y)
                    y_pred = model.predict(X)
                    r2 = r2_score(y, y_pred)

                    n = len(X)
                    p = len(selected_x_vars)
                    se = np.sqrt(np.sum((y - y_pred)**2) / (n - p - 1))

                    # Save to session and disk
                    st.session_state['model_data'] = {
                        'model': model,
                        'vars': selected_x_vars,
                        'se': se,
                        'r2': r2
                    }
                    with open("saved_baseline.pkl", "wb") as f:
                        pickle.dump(st.session_state['model_data'], f)

                    st.success(f"✅ Model trained & saved to server! R² = {r2:.4f}")
                    st.balloons()
                else:
                    st.warning("Please select both Y and at least one X variable.")
        st.markdown("---")

    # ========== ALL USERS: STEP 2 ==========
    st.header("📈 Step 2: Reporting Period Analysis")
    with st.expander("ℹ️ Need Help?"):
        st.markdown("""
        1. **Admin** must upload and train a baseline CSV in Step 1.  
        2. Upload a **Reporting CSV** with the same X columns + the actual energy column.  
        3. Select the actual energy column and click **Calculate Energy Savings**.  
        4. The tool will compute total savings and show an interactive chart.
        """)

    if not model_ready:
        st.info("⚠️ Waiting for Admin to setup the Baseline Model.")
    else:
        st.success("✅ Baseline Model is Active and Ready.")
        rep_file = st.file_uploader("Upload Reporting CSV", type=['csv'], key="rep")

        if rep_file:
            df_reporting = pd.read_csv(rep_file)
            st.write("Preview:", df_reporting.head(3))

            y_col = st.selectbox("Select Actual Energy Column (Y):", list(df_reporting.columns))

            if st.button("💰 Calculate Energy Savings", type="primary"):
                model_data = st.session_state['model_data']
                x_vars = model_data['vars']

                missing = [col for col in x_vars if col not in df_reporting.columns]
                if missing:
                    st.error(f"❌ Missing required columns: {missing}")
                else:
                    clean_rep = df_reporting[x_vars + [y_col]].dropna()
                    X_rep = clean_rep[x_vars]
                    y_actual = clean_rep[y_col]

                    adjusted_baseline = model_data['model'].predict(X_rep)
                    total_savings = (adjusted_baseline - y_actual).sum()

                    st.subheader(f"💰 Total Savings: {total_savings:,.2f} kWh")

                    # Interactive Plotly chart
                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                        x=clean_rep.index, y=adjusted_baseline,
                        mode='lines', name='Adjusted Baseline',
                        line=dict(dash='dash', color='orange')
                    ))

                    fig.add_trace(go.Scatter(
                        x=clean_rep.index, y=y_actual,
                        mode='lines', name='Actual Energy',
                        line=dict(color='blue'),
                        fill='tonexty',
                        fillcolor='rgba(144,238,144,0.3)'
                    ))

                    fig.update_layout(
                        title='IPMVP Option C: Reporting Period Savings',
                        xaxis_title='Data Point Index',
                        yaxis_title='Energy (kWh)',
                        hovermode='x unified',
                        legend=dict(orientation='h', yanchor='bottom', y=1.02)
                    )

                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("📌 Green shading: baseline is above actual (savings). Hover over the chart for exact values.")