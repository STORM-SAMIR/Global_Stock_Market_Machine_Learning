# ============================================================
# Global Stock Market — Machine Learning Project
# Streamlit App | 10 Problem Statements
# Dataset: 90,040 Records | 200 Companies | 10 Countries
# ============================================================
# HOW TO RUN LOCALLY:
#   streamlit run app.py
#
# DATASET PATH:
#   Place 'Global_Stock_Market_Master_Dataset.xlsx' in the
#   SAME folder as this app.py file.
#   Do NOT use any local machine path (D:\... won't work on GitHub/Streamlit Cloud).
# ============================================================

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                               GradientBoostingRegressor, IsolationForest)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats
from scipy.optimize import minimize

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Global Stock Market ML",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Global Stock Market — Machine Learning Project")
st.markdown("**Dataset:** 90,040 Records | 200 Companies | 10 Countries | Jan 2023 – Mar 2026")
st.markdown("---")

# ── LOAD & CACHE DATASET ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    # Dataset must be in the same folder as app.py
    # Upload 'Global_Stock_Market_Master_Dataset.xlsx' to your GitHub repo root
    # NEW — works everywhere
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'Global_Stock_Market_Master_Dataset.xlsx')
    df.columns = ['Date', 'Country', 'Company', 'Sector', 'Sub_Sector',
                  'Open', 'High', 'Low', 'Close', 'Volume',
                  'BUY', 'SELL', 'Daily_Return', 'War_Period']
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Close'] > 0]
    df = df[df['Volume'] > 0]
    df = df.reset_index(drop=True)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Price_Range'] = df['High'] - df['Low']
    df['BuySell_Ratio'] = df['BUY'] / (df['SELL'] + 1)
    return df

with st.spinner("Loading dataset..."):
    df = load_data()

st.success(f"✅ Dataset loaded — {df.shape[0]:,} rows | {df['Company'].nunique()} companies | {df['Country'].nunique()} countries")

# ── SIDEBAR NAVIGATION ────────────────────────────────────────────────────────
st.sidebar.title("📊 Problem Statements")
ps = st.sidebar.radio("Select a Problem Statement", [
    "PS1 — Stock Price Prediction",
    "PS2 — Buy/Sell Signal Classification",
    "PS3 — Cross-Sector & Country Analysis",
    "PS4 — Investor Sentiment Analysis",
    "PS5 — Portfolio Optimization",
    "PS6 — Volatility Forecasting",
    "PS7 — Anomaly Detection",
    "PS8 — Trend Classification (Multi-class)",
    "PS9 — War Period Impact Analysis",
    "PS10 — Sector Rotation Strategy",
])

st.sidebar.markdown("---")
st.sidebar.info("📁 Dataset: `Global_Stock_Market_Master_Dataset.xlsx`\nPlace in the same folder as `app.py`.")

# =============================================================================
# PS1 — STOCK PRICE PREDICTION
# =============================================================================
if ps == "PS1 — Stock Price Prediction":
    st.header("📌 PS1 — Stock Price Prediction")
    st.markdown("**Goal:** Predict the next day closing price using past stock data")
    st.markdown("**Models:** Linear Regression, Random Forest, Gradient Boosting")

    with st.spinner("Training models..."):
        df_ps1 = df.copy().sort_values(['Company', 'Date']).reset_index(drop=True)
        df_ps1['Lag1_Close']  = df_ps1.groupby('Company')['Close'].shift(1)
        df_ps1['Lag2_Close']  = df_ps1.groupby('Company')['Close'].shift(2)
        df_ps1['Lag1_Return'] = df_ps1.groupby('Company')['Daily_Return'].shift(1)
        df_ps1['MA5']         = df_ps1.groupby('Company')['Close'].transform(lambda x: x.rolling(5).mean())
        df_ps1['MA10']        = df_ps1.groupby('Company')['Close'].transform(lambda x: x.rolling(10).mean())
        df_ps1['Next_Close']  = df_ps1.groupby('Company')['Close'].shift(-1)
        df_ps1 = df_ps1.dropna()

        features = ['Open', 'High', 'Low', 'Close', 'Volume',
                    'Lag1_Close', 'Lag2_Close', 'Lag1_Return',
                    'MA5', 'MA10', 'Price_Range', 'BuySell_Ratio']
        X = df_ps1[features]
        y = df_ps1['Next_Close']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        lr_model = LinearRegression()
        lr_model.fit(X_train_sc, y_train)
        lr_preds = lr_model.predict(X_test_sc)

        rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_sc, y_train)
        rf_preds = rf_model.predict(X_test_sc)

        gb_model = GradientBoostingRegressor(n_estimators=50, random_state=42)
        gb_model.fit(X_train_sc, y_train)
        gb_preds = gb_model.predict(X_test_sc)

    results = []
    for name, preds in [("Linear Regression", lr_preds), ("Random Forest", rf_preds), ("Gradient Boosting", gb_preds)]:
        results.append({
            'Model': name,
            'MAE': round(mean_absolute_error(y_test, preds), 2),
            'RMSE': round(np.sqrt(mean_squared_error(y_test, preds)), 2),
            'R²': round(r2_score(y_test, preds), 4)
        })

    st.subheader("Model Performance")
    st.dataframe(pd.DataFrame(results), use_container_width=True)

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=('Actual vs Predicted (Random Forest — first 200 samples)', 'R² Score Comparison'))
    fig.add_trace(go.Scatter(y=y_test.values[:200], name='Actual', line=dict(color='steelblue')), row=1, col=1)
    fig.add_trace(go.Scatter(y=rf_preds[:200], name='Predicted', line=dict(color='tomato', dash='dash')), row=1, col=1)
    fig.add_trace(go.Bar(x=[r['Model'] for r in results], y=[r['R²'] for r in results],
        marker_color=['#2196F3', '#4CAF50', '#FF9800'],
        text=[str(r['R²']) for r in results], textposition='outside'), row=1, col=2)
    fig.update_layout(height=450, template='plotly_dark', title_text='PS1 Results')
    st.plotly_chart(fig, use_container_width=True)

    imp = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
    fig2 = go.Figure(go.Bar(x=imp.index, y=imp.values, marker_color='steelblue',
        text=[f'{v:.3f}' for v in imp.values], textposition='outside'))
    fig2.update_layout(title='Feature Importance (Random Forest)', template='plotly_dark',
        xaxis_tickangle=-45, height=400)
    st.plotly_chart(fig2, use_container_width=True)


# =============================================================================
# PS2 — BUY/SELL SIGNAL CLASSIFICATION
# =============================================================================
elif ps == "PS2 — Buy/Sell Signal Classification":
    st.header("📌 PS2 — Buy/Sell Signal Classification")
    st.markdown("**Goal:** Classify each trading day as BUY (1) or SELL (0)")
    st.markdown("**Models:** Logistic Regression, Decision Tree, Random Forest")

    with st.spinner("Training classifiers..."):
        df_ps2 = df.copy().sort_values(['Company', 'Date']).reset_index(drop=True)
        df_ps2['Next_Close']  = df_ps2.groupby('Company')['Close'].shift(-1)
        df_ps2['Signal']      = (df_ps2['Next_Close'] > df_ps2['Close']).astype(int)
        df_ps2['Lag1_Close']  = df_ps2.groupby('Company')['Close'].shift(1)
        df_ps2['Lag1_Return'] = df_ps2.groupby('Company')['Daily_Return'].shift(1)
        df_ps2['MA5']         = df_ps2.groupby('Company')['Close'].transform(lambda x: x.rolling(5).mean())
        df_ps2['MA10']        = df_ps2.groupby('Company')['Close'].transform(lambda x: x.rolling(10).mean())
        df_ps2 = df_ps2.dropna()

        feat_ps2 = ['Open', 'High', 'Low', 'Close', 'Volume',
                    'Lag1_Close', 'Lag1_Return', 'MA5', 'MA10',
                    'Price_Range', 'BuySell_Ratio', 'Daily_Return']
        X2 = df_ps2[feat_ps2]
        y2 = df_ps2['Signal']
        X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, shuffle=False)
        sc2 = StandardScaler()
        X2_tr = sc2.fit_transform(X2_train)
        X2_te = sc2.transform(X2_test)

        lr_clf = LogisticRegression(max_iter=500, random_state=42)
        lr_clf.fit(X2_tr, y2_train)
        dt_clf = DecisionTreeClassifier(max_depth=6, random_state=42)
        dt_clf.fit(X2_tr, y2_train)
        rf_clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rf_clf.fit(X2_tr, y2_train)

    fig = make_subplots(rows=1, cols=3,
        subplot_titles=['Logistic Reg', 'Decision Tree', 'Random Forest'])
    for i, (name, clf) in enumerate([('LR', lr_clf), ('DT', dt_clf), ('RF', rf_clf)], 1):
        preds = clf.predict(X2_te)
        cm = confusion_matrix(y2_test, preds)
        fig.add_trace(go.Heatmap(z=cm, x=['SELL','BUY'], y=['SELL','BUY'],
            colorscale='Blues', showscale=False,
            text=cm, texttemplate='%{text}', textfont_size=14), row=1, col=i)
    fig.update_layout(height=400, template='plotly_dark', title_text='PS2 — Confusion Matrices')
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    for name, clf, color in [('Logistic', lr_clf, 'blue'), ('Dec Tree', dt_clf, 'green'), ('Random Forest', rf_clf, 'orange')]:
        prob = clf.predict_proba(X2_te)[:, 1]
        fpr, tpr, _ = roc_curve(y2_test, prob)
        auc = roc_auc_score(y2_test, prob)
        fig2.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{name} (AUC={auc:.3f})'))
    fig2.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash', color='gray')))
    fig2.update_layout(title='PS2 — ROC Curve', template='plotly_dark', height=450,
        xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    st.plotly_chart(fig2, use_container_width=True)


# =============================================================================
# PS3 — CROSS-SECTOR & COUNTRY ANALYSIS
# =============================================================================
elif ps == "PS3 — Cross-Sector & Country Analysis":
    st.header("📌 PS3 — Cross-Sector & Cross-Country Market Analysis")
    st.markdown("**Goal:** Understand how different sectors perform across countries, cluster similar markets")

    country_sector = df.groupby(['Country', 'Sector']).agg(
        Avg_Return=('Daily_Return', 'mean'),
        Volatility=('Daily_Return', 'std'),
        Avg_Volume=('Volume', 'mean'),
        Avg_Close=('Close', 'mean')
    ).reset_index().dropna()

    pivot = country_sector.pivot_table(values='Avg_Return', index='Country', columns='Sector', fill_value=0)
    fig = go.Figure(go.Heatmap(z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale='RdYlGn', zmid=0,
        text=pivot.round(2).values, texttemplate='%{text}', textfont_size=7))
    fig.update_layout(title='Average Daily Return (%) by Country & Sector',
        template='plotly_dark', height=500, xaxis_tickangle=-40)
    st.plotly_chart(fig, use_container_width=True)

    feat_c = ['Avg_Return', 'Volatility', 'Avg_Volume', 'Avg_Close']
    X_c = country_sector[feat_c].copy()
    sc3 = StandardScaler()
    X_cs = sc3.fit_transform(X_c)
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    country_sector['Cluster'] = km.fit_predict(X_cs)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_cs)
    country_sector['PCA1'] = coords[:, 0]
    country_sector['PCA2'] = coords[:, 1]
    country_sector['Label'] = country_sector.apply(lambda r: f"{r['Country'][:3]}-{r['Sector'][:4]}", axis=1)

    colors = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A']
    fig2 = go.Figure()
    for c in range(4):
        sub = country_sector[country_sector['Cluster'] == c]
        fig2.add_trace(go.Scatter(x=sub['PCA1'], y=sub['PCA2'], mode='markers+text',
            name=f'Cluster {c}', marker=dict(color=colors[c], size=10),
            text=sub['Label'], textposition='top center', textfont_size=6))
    fig2.update_layout(title='K-Means Clustering (PCA view)', template='plotly_dark', height=550)
    st.plotly_chart(fig2, use_container_width=True)


# =============================================================================
# PS4 — INVESTOR SENTIMENT ANALYSIS
# =============================================================================
elif ps == "PS4 — Investor Sentiment Analysis":
    st.header("📌 PS4 — Investor Behavior & Market Sentiment Analysis")
    st.markdown("**Goal:** See if BUY/SELL investor flow data can predict future price movements")

    with st.spinner("Building sentiment model..."):
        df_ps4 = df.copy().sort_values(['Company', 'Date']).reset_index(drop=True)
        df_ps4['Net_Flow']      = df_ps4['BUY'] - df_ps4['SELL']
        df_ps4['Flow_MA5']      = df_ps4.groupby('Company')['Net_Flow'].transform(lambda x: x.rolling(5).mean())
        df_ps4['BuySell_MA5']   = df_ps4.groupby('Company')['BuySell_Ratio'].transform(lambda x: x.rolling(5).mean())
        df_ps4['Lag1_NetFlow']  = df_ps4.groupby('Company')['Net_Flow'].shift(1)
        df_ps4['Lag2_NetFlow']  = df_ps4.groupby('Company')['Net_Flow'].shift(2)
        df_ps4['Lag1_BuySell']  = df_ps4.groupby('Company')['BuySell_Ratio'].shift(1)
        df_ps4['Lag2_BuySell']  = df_ps4.groupby('Company')['BuySell_Ratio'].shift(2)
        df_ps4['Future_3d_Return'] = df_ps4.groupby('Company')['Daily_Return'].transform(
            lambda x: x.rolling(3).mean().shift(-3))
        df_ps4 = df_ps4.dropna()

        sentiment_cols = ['BuySell_Ratio', 'Net_Flow', 'BuySell_MA5', 'Flow_MA5',
                          'Lag1_BuySell', 'Lag2_BuySell', 'Lag1_NetFlow', 'Lag2_NetFlow']
        corr_data = {col: df_ps4[col].corr(df_ps4['Future_3d_Return']) for col in sentiment_cols}

    st.subheader("Correlation of Sentiment Signals with Future 3-Day Return")
    corr_df = pd.DataFrame(list(corr_data.items()), columns=['Feature', 'Correlation'])
    fig = go.Figure(go.Bar(x=corr_df['Feature'], y=corr_df['Correlation'],
        marker_color=['#4CAF50' if v > 0 else '#F44336' for v in corr_df['Correlation']]))
    fig.update_layout(template='plotly_dark', height=400, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    top_co = df_ps4['Company'].value_counts().index[0]
    co_df = df_ps4[df_ps4['Company'] == top_co].copy()
    rolling_corr = co_df['BuySell_Ratio'].rolling(30).corr(co_df['Future_3d_Return'])

    fig2 = make_subplots(rows=3, cols=1, shared_xaxes=True,
        subplot_titles=('Stock Close Price', 'BUY/SELL Ratio', '30-Day Rolling Correlation'))
    fig2.add_trace(go.Scatter(x=co_df['Date'], y=co_df['Close'], name='Close', line=dict(color='steelblue')), row=1, col=1)
    fig2.add_trace(go.Bar(x=co_df['Date'], y=co_df['BuySell_Ratio'], name='BuySell', marker_color='#4CAF50'), row=2, col=1)
    fig2.add_trace(go.Scatter(x=co_df['Date'], y=rolling_corr, fill='tozeroy',
        fillcolor='rgba(255,140,0,0.2)', line=dict(color='darkorange'), name='Rolling Corr'), row=3, col=1)
    fig2.update_layout(height=700, template='plotly_dark', title_text=f'Sentiment vs Future Return ({top_co})')
    st.plotly_chart(fig2, use_container_width=True)


# =============================================================================
# PS5 — PORTFOLIO OPTIMIZATION
# =============================================================================
elif ps == "PS5 — Portfolio Optimization":
    st.header("📌 PS5 — Portfolio Optimization (Modern Portfolio Theory)")
    st.markdown("**Goal:** Build the best possible investment portfolio — max return, min risk")
    st.markdown("**Method:** Monte Carlo Simulation, Efficient Frontier, Sharpe Ratio")

    with st.spinner("Running Monte Carlo simulation..."):
        top_companies = df['Company'].value_counts().head(20).index.tolist()
        df_ps5 = df[df['Company'].isin(top_companies)].copy()
        price_pivot = df_ps5.pivot_table(index='Date', columns='Company', values='Close')
        price_pivot = price_pivot.ffill().bfill()
        daily_returns = price_pivot.pct_change().dropna()

        n_td = 252
        exp_ret = daily_returns.mean() * n_td
        cov_mat = daily_returns.cov() * n_td
        n_assets = len(top_companies)

        np.random.seed(42)
        n_sim = 2000
        sim_ret = np.zeros(n_sim)
        sim_risk = np.zeros(n_sim)
        sim_sharpe = np.zeros(n_sim)
        sim_w = np.zeros((n_sim, n_assets))

        for i in range(n_sim):
            w = np.random.dirichlet(np.ones(n_assets))
            pr = np.dot(w, exp_ret.values)
            pk = np.sqrt(np.dot(w.T, np.dot(cov_mat.values, w)))
            sim_ret[i] = pr
            sim_risk[i] = pk
            sim_sharpe[i] = (pr - 0.02) / pk
            sim_w[i] = w

    max_idx = np.argmax(sim_sharpe)
    min_idx = np.argmin(sim_risk)

    col1, col2 = st.columns(2)
    col1.metric("Max Sharpe Return", f"{sim_ret[max_idx]:.2%}")
    col1.metric("Max Sharpe Risk",   f"{sim_risk[max_idx]:.2%}")
    col1.metric("Sharpe Ratio",      f"{sim_sharpe[max_idx]:.4f}")
    col2.metric("Min Risk Return",   f"{sim_ret[min_idx]:.2%}")
    col2.metric("Min Risk Level",    f"{sim_risk[min_idx]:.2%}")
    col2.metric("Min Risk Sharpe",   f"{sim_sharpe[min_idx]:.4f}")

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=('Efficient Frontier', 'Max Sharpe — Allocation'),
        specs=[[{'type': 'scatter'}, {'type': 'pie'}]])
    fig.add_trace(go.Scatter(x=sim_risk, y=sim_ret, mode='markers',
        marker=dict(color=sim_sharpe, colorscale='Viridis', size=3, opacity=0.5), name='Portfolios'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[sim_risk[max_idx]], y=[sim_ret[max_idx]], mode='markers', name='Max Sharpe',
        marker=dict(symbol='star', color='gold', size=18)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[sim_risk[min_idx]], y=[sim_ret[min_idx]], mode='markers', name='Min Risk',
        marker=dict(symbol='diamond', color='red', size=14)), row=1, col=1)

    opt_w = sim_w[max_idx]
    sorted_idx = np.argsort(opt_w)[::-1][:8]
    other = 1 - opt_w[sorted_idx].sum()
    labels = [top_companies[i] for i in sorted_idx] + (['Others'] if other > 0.001 else [])
    sizes  = list(opt_w[sorted_idx]) + ([other] if other > 0.001 else [])
    fig.add_trace(go.Pie(labels=labels, values=sizes, hole=0.3), row=1, col=2)
    fig.update_layout(height=500, template='plotly_dark', title_text='PS5 — Portfolio Optimization')
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PS6 — VOLATILITY FORECASTING
# =============================================================================
elif ps == "PS6 — Volatility Forecasting":
    st.header("📌 PS6 — Stock Volatility Forecasting")
    st.markdown("**Goal:** Predict how volatile (risky) a stock will be in the next 7 days")
    st.markdown("**Models:** Ridge Regression, Random Forest")

    with st.spinner("Engineering features and training models..."):
        df6 = df.copy().sort_values(['Company', 'Date']).reset_index(drop=True)
        df6['Vol_7d_Future'] = df6.groupby('Company')['Daily_Return'].transform(
            lambda x: x.shift(-1).rolling(7).std())
        df6['Vol_7d_Past']   = df6.groupby('Company')['Daily_Return'].transform(lambda x: x.rolling(7).std())
        df6['Vol_14d']       = df6.groupby('Company')['Daily_Return'].transform(lambda x: x.rolling(14).std())
        df6['Vol_30d']       = df6.groupby('Company')['Daily_Return'].transform(lambda x: x.rolling(30).std())
        df6['Avg_Range_7d']  = df6.groupby('Company')['Price_Range'].transform(lambda x: x.rolling(7).mean())
        df6['Avg_Return_7d'] = df6.groupby('Company')['Daily_Return'].transform(lambda x: x.rolling(7).mean())
        df6['Lag1_Vol']      = df6.groupby('Company')['Vol_7d_Past'].shift(1)
        df6 = df6.dropna()

        feat6 = ['Vol_7d_Past', 'Vol_14d', 'Vol_30d',
                 'Avg_Range_7d', 'Avg_Return_7d', 'Lag1_Vol', 'BuySell_Ratio', 'Close', 'Volume']
        X6 = df6[feat6]
        y6 = df6['Vol_7d_Future']
        X6_tr, X6_te, y6_tr, y6_te = train_test_split(X6, y6, test_size=0.2, shuffle=False)
        sc6 = StandardScaler()
        X6_tr_sc = sc6.fit_transform(X6_tr)
        X6_te_sc = sc6.transform(X6_te)

        ridge = Ridge(alpha=1.0)
        ridge.fit(X6_tr_sc, y6_tr)
        ridge_preds = ridge.predict(X6_te_sc)

        rf6 = RandomForestRegressor(n_estimators=80, max_depth=10, random_state=42, n_jobs=-1)
        rf6.fit(X6_tr_sc, y6_tr)
        rf6_preds = rf6.predict(X6_te_sc)

    col1, col2 = st.columns(2)
    col1.metric("Ridge MAE", f"{mean_absolute_error(y6_te, ridge_preds):.4f}")
    col1.metric("Ridge R²",  f"{r2_score(y6_te, ridge_preds):.4f}")
    col2.metric("RF MAE",    f"{mean_absolute_error(y6_te, rf6_preds):.4f}")
    col2.metric("RF R²",     f"{r2_score(y6_te, rf6_preds):.4f}")

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=('Actual vs Predicted (RF)', 'Feature Importance'))
    fig.add_trace(go.Scatter(y=y6_te.values[:150], name='Actual', line=dict(color='steelblue')), row=1, col=1)
    fig.add_trace(go.Scatter(y=rf6_preds[:150], name='Predicted', line=dict(color='tomato', dash='dash')), row=1, col=1)
    imp6 = pd.Series(rf6.feature_importances_, index=feat6).sort_values()
    fig.add_trace(go.Bar(x=imp6.values, y=imp6.index, orientation='h', marker_color='steelblue'), row=1, col=2)
    fig.update_layout(height=450, template='plotly_dark', title_text='PS6 — Volatility Forecasting')
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PS7 — ANOMALY DETECTION
# =============================================================================
elif ps == "PS7 — Anomaly Detection":
    st.header("📌 PS7 — Anomaly Detection in Stock Market Data")
    st.markdown("**Goal:** Find unusual/suspicious trading days")
    st.markdown("**Models:** Isolation Forest, Local Outlier Factor")

    with st.spinner("Running anomaly detection..."):
        df7 = df.copy()
        df7['Vol_Change'] = df7.groupby('Company')['Volume'].pct_change()
        df7['Return_Abs'] = df7['Daily_Return'].abs()
        df7 = df7.dropna()

        feat7 = ['Daily_Return', 'Volume', 'Price_Range', 'BuySell_Ratio', 'Vol_Change', 'Return_Abs', 'Close']
        X7 = df7[feat7].dropna()
        df7c = df7.loc[X7.index].copy()

        iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        df7c['IF_Anomaly'] = (iso.fit_predict(X7) == -1)
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
        df7c['LOF_Anomaly'] = (lof.fit_predict(X7) == -1)

    col1, col2 = st.columns(2)
    col1.metric("Isolation Forest Anomalies", int(df7c['IF_Anomaly'].sum()))
    col2.metric("LOF Anomalies",              int(df7c['LOF_Anomaly'].sum()))

    color_map = df7c['IF_Anomaly'].map({True: 'red', False: 'steelblue'})
    fig = go.Figure(go.Scatter(x=df7c.index, y=df7c['Daily_Return'], mode='markers',
        marker=dict(color=color_map, size=3, opacity=0.5)))
    fig.update_layout(title='Anomalies in Daily Return (red = anomaly)', template='plotly_dark', height=400)
    st.plotly_chart(fig, use_container_width=True)

    by_country = df7c.groupby('Country')['IF_Anomaly'].sum().reset_index()
    fig2 = go.Figure(go.Bar(x=by_country['Country'], y=by_country['IF_Anomaly'], marker_color='tomato'))
    fig2.update_layout(title='Anomaly Count by Country', template='plotly_dark', height=350)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Sample Anomalous Days")
    st.dataframe(df7c[df7c['IF_Anomaly']][['Date', 'Company', 'Country', 'Daily_Return', 'Volume']].head(10))


# =============================================================================
# PS8 — TREND CLASSIFICATION (MULTI-CLASS)
# =============================================================================
elif ps == "PS8 — Trend Classification (Multi-class)":
    st.header("📌 PS8 — Stock Market Trend Classification")
    st.markdown("**Goal:** Classify each stock's weekly trend as BULLISH, SIDEWAYS, or BEARISH")
    st.markdown("**Models:** Random Forest, KNN, Decision Tree")

    with st.spinner("Training multi-class classifiers..."):
        df8 = df.copy().sort_values(['Company', 'Date']).reset_index(drop=True)
        df8['Forward_5d_Return'] = df8.groupby('Company')['Daily_Return'].transform(
            lambda x: x.shift(-1).rolling(5).mean())
        df8['Trend_Label'] = df8['Forward_5d_Return'].apply(
            lambda r: 2 if r > 1.0 else (0 if r < -1.0 else 1))
        df8['MA5']       = df8.groupby('Company')['Close'].transform(lambda x: x.rolling(5).mean())
        df8['MA20']      = df8.groupby('Company')['Close'].transform(lambda x: x.rolling(20).mean())
        df8['MA_Cross']  = df8['MA5'] - df8['MA20']
        df8['Volatility'] = df8.groupby('Company')['Daily_Return'].transform(lambda x: x.rolling(10).std())
        df8['Vol_Trend'] = df8.groupby('Company')['Volume'].transform(lambda x: x.pct_change().rolling(5).mean())
        df8 = df8.dropna()

        feat8 = ['Close', 'Volume', 'MA5', 'MA20', 'MA_Cross', 'Volatility',
                 'Vol_Trend', 'Price_Range', 'BuySell_Ratio', 'Daily_Return']
        X8 = df8[feat8]
        y8 = df8['Trend_Label']
        X8_tr, X8_te, y8_tr, y8_te = train_test_split(X8, y8, test_size=0.2, shuffle=False)
        sc8 = StandardScaler()
        X8_tr_sc = sc8.fit_transform(X8_tr)
        X8_te_sc = sc8.transform(X8_te)

        rf8 = RandomForestClassifier(n_estimators=80, max_depth=10, random_state=42, n_jobs=-1)
        rf8.fit(X8_tr_sc, y8_tr)
        knn8 = KNeighborsClassifier(n_neighbors=15)
        knn8.fit(X8_tr_sc, y8_tr)
        dt8 = DecisionTreeClassifier(max_depth=8, random_state=42)
        dt8.fit(X8_tr_sc, y8_tr)

    target_names = ['BEARISH', 'SIDEWAYS', 'BULLISH']
    col1, col2, col3 = st.columns(3)
    for col, name, clf in zip([col1, col2, col3],
                               ['Random Forest', 'KNN', 'Decision Tree'],
                               [rf8, knn8, dt8]):
        acc = (clf.predict(X8_te_sc) == y8_te).mean()
        col.metric(f"{name} Accuracy", f"{acc:.4f}")

    fig = make_subplots(rows=1, cols=3, subplot_titles=['Random Forest', 'KNN', 'Decision Tree'])
    for i, clf in enumerate([rf8, knn8, dt8], 1):
        cm = confusion_matrix(y8_te, clf.predict(X8_te_sc))
        fig.add_trace(go.Heatmap(z=cm, x=target_names, y=target_names,
            colorscale='Blues', showscale=False,
            text=cm, texttemplate='%{text}', textfont_size=12), row=1, col=i)
    fig.update_layout(height=400, template='plotly_dark', title_text='PS8 — Confusion Matrices')
    st.plotly_chart(fig, use_container_width=True)

    imp8 = pd.Series(rf8.feature_importances_, index=feat8).sort_values()
    fig2 = go.Figure(go.Bar(x=imp8.values, y=imp8.index, orientation='h', marker_color='#F4A261'))
    fig2.update_layout(title='Feature Importance (Random Forest)', template='plotly_dark', height=400)
    st.plotly_chart(fig2, use_container_width=True)


# =============================================================================
# PS9 — WAR PERIOD IMPACT ANALYSIS
# =============================================================================
elif ps == "PS9 — War Period Impact Analysis":
    st.header("📌 PS9 — War Period Impact Analysis (Statistical Testing)")
    st.markdown("**Goal:** Statistically prove whether war had a significant impact on stock returns")
    st.markdown("**Methods:** T-Test, Mann-Whitney U Test, Cohen's D Effect Size")

    pre_war  = df[df['War_Period'].str.contains('Pre',  na=False)]['Daily_Return'].dropna()
    post_war = df[df['War_Period'].str.contains('Post', na=False)]['Daily_Return'].dropna()

    t_stat, p_val_t = stats.ttest_ind(pre_war, post_war, equal_var=False)
    u_stat, p_val_u = stats.mannwhitneyu(pre_war, post_war, alternative='two-sided')
    mean_diff  = post_war.mean() - pre_war.mean()
    pooled_std = np.sqrt((pre_war.std()**2 + post_war.std()**2) / 2)
    cohens_d   = mean_diff / pooled_std

    col1, col2, col3 = st.columns(3)
    col1.metric("T-Test p-value",      f"{p_val_t:.6f}", "Significant" if p_val_t < 0.05 else "Not Significant")
    col2.metric("Mann-Whitney p-value", f"{p_val_u:.6f}", "Significant" if p_val_u < 0.05 else "Not Significant")
    col3.metric("Cohen's D",            f"{cohens_d:.4f}")

    col4, col5 = st.columns(2)
    col4.metric("Pre-War Avg Return",  f"{pre_war.mean():.4f}%")
    col5.metric("Post-War Avg Return", f"{post_war.mean():.4f}%")

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=('Return Distribution: Pre vs Post War', 'Violin Plot'))
    for period, data, color in [('Pre-War', pre_war, '#2196F3'), ('Post-War', post_war, '#FF5722')]:
        fig.add_trace(go.Box(y=data, name=period, marker_color=color, boxpoints=False), row=1, col=1)
        fig.add_trace(go.Violin(y=data, name=period, marker_color=color, showlegend=False, box_visible=True), row=1, col=2)
    fig.update_layout(height=450, template='plotly_dark', title_text='PS9 — War Period Impact')
    st.plotly_chart(fig, use_container_width=True)

    war_country = df.groupby(['Country', 'War_Period'])['Daily_Return'].mean().reset_index()
    war_country['Period'] = war_country['War_Period'].apply(lambda x: 'Post-War' if 'Post' in str(x) else 'Pre-War')
    war_pivot = war_country.pivot_table(index='Country', columns='Period', values='Daily_Return')
    fig2 = go.Figure()
    for period, color in [('Pre-War', '#2196F3'), ('Post-War', '#FF5722')]:
        if period in war_pivot.columns:
            fig2.add_trace(go.Bar(x=war_pivot.index, y=war_pivot[period], name=period, marker_color=color))
    fig2.update_layout(barmode='group', title='Avg Return by Country: Pre vs Post War',
        template='plotly_dark', height=400)
    st.plotly_chart(fig2, use_container_width=True)


# =============================================================================
# PS10 — SECTOR ROTATION STRATEGY
# =============================================================================
elif ps == "PS10 — Sector Rotation Strategy":
    st.header("📌 PS10 — Sector Rotation Strategy using Return Momentum")
    st.markdown("**Goal:** Identify which sectors are gaining momentum and build a rotation strategy")
    st.markdown("**Method:** Momentum Scoring, Rolling Return Ranking, Backtesting")

    df10 = df.copy()
    df10['YearMonth'] = df10['Date'].dt.to_period('M')
    monthly = df10.groupby(['YearMonth', 'Sector'])['Daily_Return'].mean().reset_index()
    monthly.columns = ['YearMonth', 'Sector', 'Monthly_Return']
    monthly['Date'] = monthly['YearMonth'].dt.to_timestamp()
    monthly = monthly.sort_values(['Sector', 'Date']).reset_index(drop=True)
    monthly['Momentum_3m'] = monthly.groupby('Sector')['Monthly_Return'].transform(lambda x: x.rolling(3).mean())
    monthly['Momentum_Rank'] = monthly.groupby('YearMonth')['Momentum_3m'].rank(ascending=False, method='dense')
    monthly = monthly.dropna()

    strat_rets, bench_rets, dates_list = [], [], []
    unique_months = sorted(monthly['YearMonth'].unique())
    for i in range(3, len(unique_months) - 1):
        curr, nxt = unique_months[i], unique_months[i+1]
        curr_data = monthly[monthly['YearMonth'] == curr]
        top3 = curr_data.nsmallest(3, 'Momentum_Rank')['Sector'].tolist()
        next_data = monthly[monthly['YearMonth'] == nxt]
        top3_next = next_data[next_data['Sector'].isin(top3)]['Monthly_Return']
        if len(top3_next) > 0:
            strat_rets.append(top3_next.mean())
            bench_rets.append(next_data['Monthly_Return'].mean())
            dates_list.append(curr.to_timestamp())

    strat_s = pd.Series(strat_rets, index=dates_list)
    bench_s = pd.Series(bench_rets, index=dates_list)
    cum_s   = (1 + strat_s/100).cumprod()
    cum_b   = (1 + bench_s/100).cumprod()

    col1, col2, col3 = st.columns(3)
    col1.metric("Strategy Avg Monthly Return",   f"{strat_s.mean():.4f}%")
    col2.metric("Benchmark Avg Monthly Return",  f"{bench_s.mean():.4f}%")
    col3.metric("Outperformance",                f"{(strat_s.mean()-bench_s.mean()):.4f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(cum_s.index), y=cum_s.values, name='Sector Rotation Strategy',
        line=dict(color='#4CAF50', width=2.5)))
    fig.add_trace(go.Scatter(x=list(cum_b.index), y=cum_b.values, name='Benchmark (All Sectors)',
        line=dict(color='steelblue', width=2.5, dash='dash')))
    fig.update_layout(title='PS10 — Cumulative Return: Strategy vs Benchmark',
        template='plotly_dark', height=420)
    st.plotly_chart(fig, use_container_width=True)

    pivot_hm = monthly.pivot_table(index='YearMonth', columns='Sector', values='Momentum_3m').tail(18)
    pivot_hm.index = [str(x) for x in pivot_hm.index]
    top8 = pivot_hm.std().nlargest(8).index
    pivot_hm = pivot_hm[top8]
    fig2 = go.Figure(go.Heatmap(z=pivot_hm.values,
        x=pivot_hm.columns.tolist(), y=pivot_hm.index.tolist(),
        colorscale='RdYlGn', zmid=0,
        text=pivot_hm.round(2).values, texttemplate='%{text}', textfont_size=9))
    fig2.update_layout(title='Sector Momentum Heatmap (Last 18 Months)',
        template='plotly_dark', height=500, xaxis_tickangle=-35)
    st.plotly_chart(fig2, use_container_width=True)


# ── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("**Global Stock Market ML Project** | Built with Streamlit & Scikit-learn")
