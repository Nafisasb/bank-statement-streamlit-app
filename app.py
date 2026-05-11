
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import anthropic
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

# ── Load artefacts ─────────────────────────────────────────────────────────
model      = pickle.load(open("expense_model.pkl",  "rb"))
vectorizer = pickle.load(open("vectorizer.pkl",      "rb"))

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Personal Finance Advisor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: linear-gradient(to right, #f8fafc, #e2e8f0); }
[data-testid="stSidebar"] { background-color: #0f172a; }
[data-testid="stSidebar"] * { color: white; }
h1 { color: #0f172a; font-size: 44px !important; font-weight: 800 !important; }
h2, h3 { color: #1e293b; }
.metric-card { background: white; padding: 22px; border-radius: 18px;
               box-shadow: 0 4px 12px rgba(0,0,0,.08); text-align: center; margin-bottom: 20px; }
.info-card   { background: white; padding: 26px; border-radius: 18px;
               box-shadow: 0 4px 12px rgba(0,0,0,.08); margin-bottom: 22px; }
.recommend-box { background: #dcfce7; padding: 18px; border-left: 6px solid #22c55e;
                 border-radius: 10px; margin-bottom: 15px; }
.medium-box    { background: #dbeafe; padding: 18px; border-left: 6px solid #2563eb;
                 border-radius: 10px; margin-bottom: 15px; }
.llm-box       { background: #fef9c3; padding: 22px; border-left: 6px solid #eab308;
                 border-radius: 10px; margin-bottom: 20px; white-space: pre-wrap; }
.anomaly-box   { background: #fee2e2; padding: 18px; border-left: 6px solid #ef4444;
                 border-radius: 10px; margin-bottom: 15px; }
.score-box { background: linear-gradient(to right, #2563eb, #1d4ed8); color: white;
             padding: 30px; border-radius: 20px; text-align: center; margin-bottom: 25px; }
.score-box h1, .score-box h3, .score-box p { color: white !important; }
</style>
""", unsafe_allow_html=True)

st.title("AI Personal Finance Advisor")
st.write("An explainable AI system combining ML, anomaly detection, forecasting, and LLM-powered advice.")

# ── Sidebar navigation ─────────────────────────────────────────────────────
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Analyse Spending", "LLM Financial Advice", "XAI & Model Insights",
     "Anomaly Detection", "Spending Forecast", "About Project"]
)

ANTHROPIC_API_KEY = st.sidebar.text_input(
    "Anthropic API Key", type="password",
    help="Required for LLM-powered advice. Get one at console.anthropic.com"
)

# ── Shared data loading helper ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_excel("/content/personal_transactions_dashboard_ready (2).xlsx")
    df = df.dropna(subset=["Description", "Amount"])
    df["Description"] = df["Description"].astype(str).str.lower().str.strip()
    df["Amount"]      = pd.to_numeric(df["Amount"], errors="coerce")
    df = df.dropna(subset=["Amount"])
    X_vec = vectorizer.transform(df["Description"])
    df["Predicted Category"] = model.predict(X_vec)
    # Anomaly detection
    iso = IsolationForest(contamination=0.05, random_state=42)
    df["Anomaly"] = iso.fit_predict(df[["Amount"]])
    df["Anomaly_Label"] = df["Anomaly"].map({1: "Normal", -1: "⚠️ Anomaly"})
    return df

df = load_data()
total_spending    = df["Amount"].sum()
category_spending = df.groupby("Predicted Category")["Amount"].sum().sort_values(ascending=False)
top_category      = category_spending.idxmax()
anomaly_count     = (df["Anomaly"] == -1).sum()

# ══════════════════════════════════════════════════════════════════════════════
if page == "Home":
    st.markdown("""
    <div class="info-card">
    <h2>Project Purpose</h2>
    <p>This AI-based system analyses transaction data, categorises expenses using NLP + ML,
    detects anomalous transactions, forecasts future spending, and generates personalised
    savings advice powered by a Large Language Model (Claude by Anthropic).</p>
    </div>""", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    for col, title, desc in zip(
        [col1, col2, col3, col4],
        ["NLP + ML", "Anomaly Detection", "Forecasting", "LLM Advice"],
        ["TF-IDF + Logistic Regression", "Isolation Forest", "Time-Series Regression", "Anthropic Claude API"]
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card"><h3>{title}</h3><p>{desc}</p></div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
elif page == "Analyse Spending":
    st.header("Transaction Analysis Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    for col, label, value in zip(
        [col1, col2, col3, col4],
        ["Total Transactions", "Total Spending", "Top Category", "Anomalies"],
        [len(df), f"${total_spending:,.2f}", top_category, f"{anomaly_count} flagged"]
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card"><h3>{label}</h3><h2>{value}</h2></div>
            """, unsafe_allow_html=True)

    st.subheader("AI Expense Categorisation (sample)")
    st.dataframe(df[["Description", "Amount", "Predicted Category", "Anomaly_Label"]].head(30))

    st.subheader("Spending by Category")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    category_spending.plot(kind="bar", ax=axes[0], color="#6366f1")
    axes[0].set_title("Total Spending by AI-Predicted Category")
    axes[0].set_xlabel("Category"); axes[0].set_ylabel("Amount ($)")
    axes[0].tick_params(axis="x", rotation=45)
    category_spending.head(6).plot(kind="pie", autopct="%1.1f%%", ax=axes[1])
    axes[1].set_ylabel("")
    axes[1].set_title("Top 6 Category Distribution")
    plt.tight_layout(); st.pyplot(fig)

    if "Month" in df.columns:
        st.subheader("Monthly Spending Trend")
        monthly = df.groupby("Month")["Amount"].sum()
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        monthly.plot(kind="line", marker="o", ax=ax2, color="#6366f1", linewidth=2)
        ax2.set_title("Monthly Spending Trend"); ax2.set_xlabel("Month"); ax2.set_ylabel("Amount ($)")
        plt.tight_layout(); st.pyplot(fig2)

    st.subheader("Rule-Based Savings Recommendations")
    for cat, amt in category_spending.items():
        pct = (amt / total_spending) * 100
        if pct > 20:
            st.markdown(f"""
            <div class="recommend-box"><b>High Priority — {cat}</b> ({pct:.1f}% of spending)<br>
            Reducing by 15% could save <b>${amt*0.15:,.2f}</b></div>
            """, unsafe_allow_html=True)
        elif pct > 10:
            st.markdown(f"""
            <div class="medium-box"><b>Medium Priority — {cat}</b> ({pct:.1f}% of spending)<br>
            Reducing by 10% could save <b>${amt*0.10:,.2f}</b></div>
            """, unsafe_allow_html=True)

    st.subheader("AI Financial Health Score")
    disc_cats = ["Shopping","Restaurants","Fast Food","Entertainment","Movies & Dvds","Coffee Shops"]
    disc_spend = category_spending[category_spending.index.isin(disc_cats)].sum()
    disc_ratio = (disc_spend / total_spending) * 100
    score, status = (90, "Excellent") if disc_ratio < 15 else (75, "Good") if disc_ratio < 25 else (60, "Moderate") if disc_ratio < 35 else (45, "Needs Improvement")
    st.markdown(f"""
    <div class="score-box">
    <h1>{score}/100</h1><h3>Financial Health Score</h3>
    <p>Status: {status} | Discretionary: {disc_ratio:.1f}% of total spending</p>
    </div>""", unsafe_allow_html=True)

    st.subheader("Model Performance Comparison")
    perf_df = pd.DataFrame({
        "Model"          : ["Naive Bayes", "Logistic Regression", "Random Forest"],
        "Test Accuracy"  : [0.9198, 0.9259, 0.9383],
        "CV Mean"        : [0.9141, 0.9221, 0.9310],
        "Runtime (s)"    : [0.0066, 0.0766, 0.4014]
    })
    st.dataframe(perf_df)

# ══════════════════════════════════════════════════════════════════════════════
elif page == "LLM Financial Advice":
    st.header("🤖 LLM-Powered Personalised Financial Advice")
    st.write("Uses Anthropic Claude to generate contextually aware savings advice from your spending profile.")

    summary_lines = [f"Total spending: ${total_spending:,.2f}",
                     f"Anomalies detected: {anomaly_count}",
                     "Spending by category:"]
    for cat, amt in category_spending.items():
        pct = (amt / total_spending) * 100
        summary_lines.append(f"  - {cat}: ${amt:,.2f} ({pct:.1f}%)")
    spending_summary = "\n".join(summary_lines)

    st.code(spending_summary, language="")

    if st.button("Generate LLM Advice", type="primary"):
        if not ANTHROPIC_API_KEY:
            st.error("Please enter your Anthropic API key in the sidebar.")
        else:
            with st.spinner("Calling Claude API..."):
                try:
                    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
                    message = client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=600,
                        system=(
                            "You are an expert personal finance advisor. Analyse the spending summary "
                            "and provide: 1) A brief overall assessment (2 sentences). "
                            "2) Three specific savings recommendations with estimated savings. "
                            "3) One behavioural insight. Be specific and practical."
                        ),
                        messages=[{"role": "user", "content":
                                   f"Spending summary:\n\n{spending_summary}\n\nProvide personalised financial advice."}]
                    )
                    advice = message.content[0].text
                    st.markdown(f"""
                    <div class="llm-box"><b>Claude's Personalised Advice:</b><br><br>{advice}</div>
                    """, unsafe_allow_html=True)
                    st.success("✅ Advice generated successfully via Anthropic Claude API")
                except Exception as e:
                    st.error(f"API Error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
elif page == "XAI & Model Insights":
    st.header("🔍 Explainable AI — Model Transparency")
    st.write("Logistic Regression coefficients reveal which words drive each expense category prediction.")

    feature_names = np.array(vectorizer.get_feature_names_out())
    top_cats = category_spending.head(6).index.tolist()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for idx, cat in enumerate(top_cats):
        if cat in model.classes_:
            i       = list(model.classes_).index(cat)
            top_idx = model.coef_[i].argsort()[-5:][::-1]
            tokens  = feature_names[top_idx]
            weights = model.coef_[i][top_idx]
            axes[idx].barh(tokens[::-1], weights[::-1], color="#6366f1")
            axes[idx].set_title(cat, fontweight="bold")
            axes[idx].set_xlabel("LR Weight")
    plt.suptitle("Top Predictive Tokens per Category (XAI)", fontsize=14, fontweight="bold")
    plt.tight_layout(); st.pyplot(fig)

    st.subheader("AI Insights Page")
    st.markdown("""
    <div class="info-card">
    <h3>How the System Works</h3>
    <p><b>TF-IDF Vectorisation:</b> Converts transaction descriptions into numerical feature vectors
    by weighting terms by frequency and inverse document frequency, reducing noise from common words.</p>
    <p><b>Logistic Regression Classifier:</b> Learns a linear decision boundary per category.
    Coefficients are directly interpretable as feature importances (XAI).</p>
    <p><b>Isolation Forest:</b> An ensemble of random trees that isolates anomalous transactions
    with unusually high amounts, without requiring labelled fraud data.</p>
    <p><b>LLM (Claude):</b> Aggregated spending statistics — not raw transactions — are sent to the
    Anthropic API. The LLM generates natural-language advice beyond hardcoded rules.</p>
    <p><b>Responsible AI:</b> No personal banking credentials or PII are transmitted.
    Only anonymised category-level summaries reach the API.</p>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
elif page == "Anomaly Detection":
    st.header("⚠️ Anomaly Detection — Isolation Forest")
    st.write("Flags statistically unusual transactions that may represent overspending or data errors.")

    anomalies = df[df["Anomaly"] == -1]
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<div class='metric-card'><h3>Anomalies Detected</h3><h2>{len(anomalies)}</h2></div>",
                    unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><h3>Anomaly Rate</h3><h2>{len(anomalies)/len(df)*100:.1f}%</h2></div>",
                    unsafe_allow_html=True)

    st.subheader("Flagged Transactions")
    st.dataframe(anomalies[["Description", "Predicted Category", "Amount"]]
                 .sort_values("Amount", ascending=False).head(20))

    fig, ax = plt.subplots(figsize=(12, 5))
    normal = df[df["Anomaly"] == 1]
    ax.scatter(range(len(normal)), normal["Amount"], c="#22c55e", alpha=0.4, s=15, label="Normal")
    ax.scatter(
        [df.index.get_loc(i) for i in anomalies.index],
        anomalies["Amount"], c="#ef4444", s=60, marker="x", label="Anomaly"
    )
    ax.set_title("Isolation Forest — Transaction Anomaly Detection", fontweight="bold")
    ax.set_xlabel("Transaction Index"); ax.set_ylabel("Amount ($)")
    ax.legend(); plt.tight_layout(); st.pyplot(fig)

# ══════════════════════════════════════════════════════════════════════════════
elif page == "Spending Forecast":
    st.header("📈 Spending Forecast — Time-Series Regression")

    if "Month" not in df.columns:
        st.warning("'Month' column not found in dataset. Add a numeric Month column to enable forecasting.")
    else:
        monthly = df.groupby("Month")["Amount"].sum().reset_index()
        monthly.columns = ["Month", "Total"]
        monthly = monthly.sort_values("Month").reset_index(drop=True)
        monthly["Idx"] = range(1, len(monthly) + 1)

        lr = LinearRegression()
        lr.fit(monthly[["Idx"]], monthly["Total"])
        next_m    = monthly["Idx"].max() + 1
        next_pred = lr.predict([[next_m]])[0]

        st.markdown(f"""
        <div class="metric-card"><h3>Predicted Next Month Spending</h3>
        <h2>${next_pred:,.2f}</h2></div>""", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(monthly["Idx"], monthly["Total"], marker="o", color="#6366f1",
                linewidth=2, label="Actual Spending")
        all_idx  = list(monthly["Idx"]) + [next_m]
        all_pred = lr.predict([[i] for i in all_idx])
        ax.plot(all_idx, all_pred, "--", color="#f59e0b", linewidth=2, label="Trend / Forecast")
        ax.scatter([next_m], [next_pred], color="#ef4444", s=120, zorder=5,
                   label=f"Forecast: ${next_pred:,.2f}")
        ax.set_title("Monthly Spending Trend & Next-Month Forecast", fontweight="bold")
        ax.set_xlabel("Month Index"); ax.set_ylabel("Amount ($)")
        ax.legend(); plt.tight_layout(); st.pyplot(fig)

# ══════════════════════════════════════════════════════════════════════════════
elif page == "About Project":
    st.markdown("""
    <div class="info-card">
    <h2>About This Project</h2>
    <p><b>Title:</b> Explainable AI System for Personal Expense Analysis and Smart Savings Recommendations</p>
    <p><b>AI Paradigms:</b></p>
    <ul>
      <li>NLP + Supervised ML (TF-IDF + Logistic Regression / Naive Bayes / Random Forest)</li>
      <li>Probabilistic Anomaly Detection (Isolation Forest)</li>
      <li>Time-Series Forecasting (Linear Regression)</li>
      <li>Large Language Model / Generative AI (Anthropic Claude)</li>
      <li>Explainable AI (LR coefficient-based feature importance)</li>
    </ul>
    <p><b>Responsible AI:</b> No real bank credentials are used. Only aggregated, anonymised statistics
    are transmitted to the LLM API. The system is a research prototype.</p>
    </div>""", unsafe_allow_html=True)
