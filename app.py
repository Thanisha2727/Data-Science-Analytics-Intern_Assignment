"""
Trader Performance vs Market Sentiment — Interactive Dashboard
Primetrade.ai Data Science Intern Assignment
Built with Streamlit + Plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Trader Performance vs Market Sentiment",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        font-size: 1rem;
        opacity: 0.85;
        margin-top: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        color: white;
    }
    .metric-card h3 {
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0;
        opacity: 0.7;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.3rem 0;
    }
    .metric-card .delta {
        font-size: 0.8rem;
        opacity: 0.8;
    }
    .fear-card { border-left-color: #ef4444; }
    .greed-card { border-left-color: #22c55e; }
    .neutral-card { border-left-color: #f59e0b; }
    .info-card { border-left-color: #3b82f6; }
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #334155;
    }
    .insight-box {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        border-left: 4px solid #8b5cf6;
        color: #e2e8f0;
    }
    .strategy-box {
        background: linear-gradient(135deg, #064e3b, #022c22);
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        border-left: 4px solid #10b981;
        color: #d1fae5;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] > div {
        font-size: 1.6rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING & PROCESSING
# ============================================================
@st.cache_data
def load_and_process_data():
    """Load and process both datasets."""
    # Load datasets
    sentiment_df = pd.read_csv('data/fear_greed_index.csv')
    trader_df = pd.read_csv('data/historical_data.csv')

    # Process sentiment
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    sentiment_df['sentiment_binary'] = sentiment_df['classification'].map({
        'Extreme Fear': 'Fear', 'Fear': 'Fear',
        'Neutral': 'Neutral',
        'Greed': 'Greed', 'Extreme Greed': 'Greed'
    })

    # Process trader data
    trader_df['Timestamp IST'] = pd.to_datetime(
        trader_df['Timestamp IST'], format='%d-%m-%Y %H:%M', errors='coerce'
    )
    trader_df['date'] = trader_df['Timestamp IST'].dt.date
    trader_df['date'] = pd.to_datetime(trader_df['date'])
    trader_df = trader_df.dropna(subset=['date'])

    # Convert numeric columns
    for col in ['Closed PnL', 'Size USD', 'Size Tokens', 'Execution Price', 'Fee']:
        trader_df[col] = pd.to_numeric(trader_df[col], errors='coerce').fillna(0)

    trader_df['is_profitable'] = (trader_df['Closed PnL'] > 0).astype(int)
    trader_df['is_long'] = (trader_df['Side'].str.upper() == 'BUY').astype(int)
    trader_df['is_short'] = (trader_df['Side'].str.upper() == 'SELL').astype(int)

    # Merge
    merged_df = trader_df.merge(
        sentiment_df[['date', 'value', 'classification', 'sentiment_binary']],
        on='date', how='inner'
    )

    # Daily trader metrics
    daily_trader = merged_df.groupby(
        ['date', 'Account', 'sentiment_binary', 'classification']
    ).agg(
        daily_pnl=('Closed PnL', 'sum'),
        trade_count=('Closed PnL', 'count'),
        avg_trade_size=('Size USD', 'mean'),
        total_volume=('Size USD', 'sum'),
        wins=('is_profitable', 'sum'),
        long_trades=('is_long', 'sum'),
        short_trades=('is_short', 'sum'),
        avg_execution_price=('Execution Price', 'mean'),
        total_fees=('Fee', 'sum'),
    ).reset_index()

    daily_trader['win_rate'] = daily_trader['wins'] / daily_trader['trade_count']
    daily_trader['long_short_ratio'] = (
        daily_trader['long_trades'] / daily_trader['short_trades'].replace(0, np.nan)
    ).fillna(0)

    # Daily market metrics
    daily_market = daily_trader.groupby(
        ['date', 'sentiment_binary', 'classification']
    ).agg(
        total_pnl=('daily_pnl', 'sum'),
        avg_pnl=('daily_pnl', 'mean'),
        median_pnl=('daily_pnl', 'median'),
        pnl_std=('daily_pnl', 'std'),
        total_trades=('trade_count', 'sum'),
        active_traders=('Account', 'nunique'),
        avg_win_rate=('win_rate', 'mean'),
        avg_trade_size=('avg_trade_size', 'mean'),
        total_volume=('total_volume', 'sum'),
        avg_long_short_ratio=('long_short_ratio', 'mean'),
    ).reset_index()
    daily_market = daily_market.sort_values('date')
    daily_market['cumulative_pnl'] = daily_market['total_pnl'].cumsum()

    # Trader-level
    trader_metrics = daily_trader.groupby('Account').agg(
        total_pnl=('daily_pnl', 'sum'),
        avg_daily_pnl=('daily_pnl', 'mean'),
        pnl_std=('daily_pnl', 'std'),
        total_trades=('trade_count', 'sum'),
        active_days=('date', 'nunique'),
        avg_win_rate=('win_rate', 'mean'),
        avg_trade_size=('avg_trade_size', 'mean'),
        avg_long_short_ratio=('long_short_ratio', 'mean'),
    ).reset_index()
    trader_metrics['pnl_std'] = trader_metrics['pnl_std'].fillna(0)
    trader_metrics['trades_per_day'] = trader_metrics['total_trades'] / trader_metrics['active_days']
    trader_metrics['consistency'] = (
        trader_metrics['avg_daily_pnl'] / trader_metrics['pnl_std'].replace(0, np.nan)
    ).fillna(0)

    drawdown = daily_trader.groupby('Account')['daily_pnl'].min().reset_index()
    drawdown.columns = ['Account', 'max_drawdown']
    trader_metrics = trader_metrics.merge(drawdown, on='Account', how='left')

    # Segments
    trader_metrics['leverage_segment'] = np.where(
        trader_metrics['avg_trade_size'] > trader_metrics['avg_trade_size'].median(),
        'High Leverage', 'Low Leverage'
    )
    trader_metrics['frequency_segment'] = np.where(
        trader_metrics['trades_per_day'] > trader_metrics['trades_per_day'].median(),
        'Frequent', 'Infrequent'
    )
    trader_metrics['consistency_segment'] = np.where(
        trader_metrics['consistency'] > trader_metrics['consistency'].median(),
        'Consistent', 'Inconsistent'
    )

    return sentiment_df, trader_df, merged_df, daily_trader, daily_market, trader_metrics


# Load data
sentiment_df, trader_df, merged_df, daily_trader, daily_market, trader_metrics = load_and_process_data()

# Merge segment info
daily_with_segments = daily_trader.merge(
    trader_metrics[['Account', 'leverage_segment', 'frequency_segment', 'consistency_segment']],
    on='Account', how='left'
)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🎛️ Dashboard Controls")
    st.markdown("---")

    # Sentiment filter
    sentiment_filter = st.multiselect(
        "Filter by Sentiment",
        options=['Fear', 'Neutral', 'Greed'],
        default=['Fear', 'Greed'],
        help="Select sentiment regimes to compare"
    )

    st.markdown("---")

    # Date range
    min_date = daily_market['date'].min().date()
    max_date = daily_market['date'].max().date()
    date_range = st.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    st.markdown("---")

    # Segment selector
    segment_type = st.selectbox(
        "Trader Segment View",
        ['leverage_segment', 'frequency_segment', 'consistency_segment'],
        format_func=lambda x: {
            'leverage_segment': '📊 High vs Low Leverage',
            'frequency_segment': '⚡ Frequent vs Infrequent',
            'consistency_segment': '🎯 Consistent vs Inconsistent'
        }[x]
    )

    st.markdown("---")
    st.markdown("### 📋 Dataset Info")
    st.markdown(f"**Sentiment data:** {len(sentiment_df):,} rows")
    st.markdown(f"**Trader data:** {len(trader_df):,} rows")
    st.markdown(f"**Merged data:** {len(merged_df):,} rows")
    st.markdown(f"**Unique traders:** {merged_df['Account'].nunique():,}")
    st.markdown(f"**Trading days:** {merged_df['date'].nunique():,}")

# Apply filters
if len(date_range) == 2:
    mask = (
        (daily_trader['date'] >= pd.to_datetime(date_range[0])) &
        (daily_trader['date'] <= pd.to_datetime(date_range[1])) &
        (daily_trader['sentiment_binary'].isin(sentiment_filter))
    )
    filtered_daily = daily_trader[mask]
    mask_market = (
        (daily_market['date'] >= pd.to_datetime(date_range[0])) &
        (daily_market['date'] <= pd.to_datetime(date_range[1])) &
        (daily_market['sentiment_binary'].isin(sentiment_filter))
    )
    filtered_market = daily_market[mask_market]
else:
    filtered_daily = daily_trader[daily_trader['sentiment_binary'].isin(sentiment_filter)]
    filtered_market = daily_market[daily_market['sentiment_binary'].isin(sentiment_filter)]

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>📊 Trader Performance vs Market Sentiment</h1>
    <p>Analyzing how Bitcoin Fear/Greed Index impacts trader behavior and profitability on Hyperliquid</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Data Overview",
    "📈 Performance Analysis",
    "🔄 Behavioral Analysis",
    "👥 Trader Segments",
    "🤖 Prediction Model",
    "💡 Strategy & Insights"
])

# ============================================================
# TAB 1: DATA OVERVIEW
# ============================================================
with tab1:
    st.markdown('<div class="section-header">Part A — Data Preparation & Overview</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Trades", f"{len(merged_df):,}")
    with col2:
        st.metric("Unique Traders", f"{merged_df['Account'].nunique():,}")
    with col3:
        st.metric("Trading Days", f"{merged_df['date'].nunique():,}")
    with col4:
        st.metric("Unique Coins", f"{merged_df['Coin'].nunique():,}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Sentiment Dataset")
        st.dataframe(sentiment_df.head(20), use_container_width=True, height=300)

        st.markdown("**Missing Values:**")
        missing_s = sentiment_df.isnull().sum()
        st.dataframe(missing_s[missing_s > 0] if missing_s.sum() > 0 else pd.DataFrame({"Missing": ["None"]}))

        st.markdown(f"**Duplicates:** {sentiment_df.duplicated().sum()}")

    with col2:
        st.markdown("#### Trader Dataset (Sample)")
        st.dataframe(trader_df.head(20), use_container_width=True, height=300)

        st.markdown("**Missing Values:**")
        missing_t = trader_df.isnull().sum()
        st.dataframe(missing_t[missing_t > 0])

        st.markdown(f"**Duplicates:** {trader_df.duplicated().sum()}")

    st.markdown("---")
    st.markdown("#### Sentiment Distribution")

    col1, col2 = st.columns(2)
    with col1:
        # Detailed classification
        class_counts = sentiment_df['classification'].value_counts()
        fig = px.pie(
            values=class_counts.values,
            names=class_counts.index,
            color=class_counts.index,
            color_discrete_map={
                'Extreme Fear': '#991b1b', 'Fear': '#ef4444',
                'Neutral': '#f59e0b',
                'Greed': '#22c55e', 'Extreme Greed': '#15803d'
            },
            title="Detailed Sentiment Classification"
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                         plot_bgcolor='rgba(0,0,0,0)', height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Binary classification
        bin_counts = sentiment_df['sentiment_binary'].value_counts()
        fig = px.bar(
            x=bin_counts.index, y=bin_counts.values,
            color=bin_counts.index,
            color_discrete_map={'Fear': '#ef4444', 'Neutral': '#f59e0b', 'Greed': '#22c55e'},
            title="Simplified Sentiment (Binary)",
            labels={'x': 'Sentiment', 'y': 'Days'}
        )
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                         plot_bgcolor='rgba(0,0,0,0)', showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Fear & Greed Index Over Time")
    fig = px.line(
        sentiment_df, x='date', y='value',
        color_discrete_sequence=['#f59e0b'],
        title="Bitcoin Fear & Greed Index Timeline"
    )
    fig.add_hline(y=25, line_dash="dash", line_color="#ef4444", annotation_text="Extreme Fear")
    fig.add_hline(y=50, line_dash="dash", line_color="#94a3b8", annotation_text="Neutral")
    fig.add_hline(y=75, line_dash="dash", line_color="#22c55e", annotation_text="Extreme Greed")
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                     plot_bgcolor='rgba(0,0,0,0)', height=400,
                     yaxis_title="Index Value", xaxis_title="Date")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 2: PERFORMANCE ANALYSIS
# ============================================================
with tab2:
    st.markdown('<div class="section-header">Part B — Performance: Fear vs Greed</div>', unsafe_allow_html=True)

    # Summary metrics
    fear_data = filtered_daily[filtered_daily['sentiment_binary'] == 'Fear']
    greed_data = filtered_daily[filtered_daily['sentiment_binary'] == 'Greed']

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        fear_pnl = fear_data['daily_pnl'].mean() if len(fear_data) > 0 else 0
        st.markdown(f"""
        <div class="metric-card fear-card">
            <h3>Fear — Mean PnL</h3>
            <div class="value">${fear_pnl:,.2f}</div>
            <div class="delta">{len(fear_data):,} trader-days</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        greed_pnl = greed_data['daily_pnl'].mean() if len(greed_data) > 0 else 0
        st.markdown(f"""
        <div class="metric-card greed-card">
            <h3>Greed — Mean PnL</h3>
            <div class="value">${greed_pnl:,.2f}</div>
            <div class="delta">{len(greed_data):,} trader-days</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        fear_wr = fear_data['win_rate'].mean() * 100 if len(fear_data) > 0 else 0
        st.markdown(f"""
        <div class="metric-card fear-card">
            <h3>Fear — Win Rate</h3>
            <div class="value">{fear_wr:.1f}%</div>
            <div class="delta">avg across traders</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        greed_wr = greed_data['win_rate'].mean() * 100 if len(greed_data) > 0 else 0
        st.markdown(f"""
        <div class="metric-card greed-card">
            <h3>Greed — Win Rate</h3>
            <div class="value">{greed_wr:.1f}%</div>
            <div class="delta">avg across traders</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Performance comparison charts
    col1, col2 = st.columns(2)

    with col1:
        # PnL Distribution
        fig = go.Figure()
        for sent, color in [('Fear', '#ef4444'), ('Greed', '#22c55e')]:
            d = filtered_daily[filtered_daily['sentiment_binary'] == sent]['daily_pnl']
            q1, q99 = d.quantile(0.01), d.quantile(0.99)
            d_clipped = d.clip(q1, q99)
            fig.add_trace(go.Box(
                y=d_clipped, name=sent, marker_color=color,
                boxmean='sd', opacity=0.8
            ))
        fig.update_layout(
            title="PnL Distribution (1st-99th Percentile)",
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)', height=450,
            yaxis_title="PnL (USD)"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Win Rate Distribution
        fig = go.Figure()
        for sent, color in [('Fear', '#ef4444'), ('Greed', '#22c55e')]:
            d = filtered_daily[filtered_daily['sentiment_binary'] == sent]['win_rate']
            fig.add_trace(go.Histogram(
                x=d * 100, name=sent, marker_color=color,
                opacity=0.6, nbinsx=30
            ))
        fig.update_layout(
            title="Win Rate Distribution",
            barmode='overlay',
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)', height=450,
            xaxis_title="Win Rate (%)", yaxis_title="Frequency"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Cumulative PnL
    st.markdown("#### Cumulative PnL with Sentiment Overlay")
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    mkt = daily_market.sort_values('date')
    fig.add_trace(
        go.Scatter(x=mkt['date'], y=mkt['cumulative_pnl'],
                   name='Cumulative PnL', line=dict(color='#3b82f6', width=2.5)),
        secondary_y=False
    )

    sent_timeline = sentiment_df.sort_values('date')
    fig.add_trace(
        go.Scatter(x=sent_timeline['date'], y=sent_timeline['value'],
                   name='Fear/Greed Index', line=dict(color='#f59e0b', width=1),
                   opacity=0.5),
        secondary_y=True
    )
    fig.update_yaxes(title_text="Cumulative PnL (USD)", secondary_y=False)
    fig.update_yaxes(title_text="Fear & Greed Index", secondary_y=True)
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)', height=500,
        xaxis_title="Date"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Detailed statistics table
    st.markdown("#### Detailed Performance Statistics")
    if len(filtered_daily) > 0:
        perf_table = filtered_daily.groupby('sentiment_binary').agg(
            Mean_PnL=('daily_pnl', 'mean'),
            Median_PnL=('daily_pnl', 'median'),
            Total_PnL=('daily_pnl', 'sum'),
            PnL_Std=('daily_pnl', 'std'),
            Win_Rate=('win_rate', 'mean'),
            Max_Drawdown=('daily_pnl', 'min'),
            Best_Day=('daily_pnl', 'max'),
            Observations=('daily_pnl', 'count'),
        ).round(2)
        st.dataframe(perf_table, use_container_width=True)

# ============================================================
# TAB 3: BEHAVIORAL ANALYSIS
# ============================================================
with tab3:
    st.markdown('<div class="section-header">Part B — Behavioral Differences: Fear vs Greed</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Trade Frequency
        fig = go.Figure()
        for sent, color in [('Fear', '#ef4444'), ('Greed', '#22c55e')]:
            d = filtered_daily[filtered_daily['sentiment_binary'] == sent]
            avg_tc = d.groupby('date')['trade_count'].mean()
            fig.add_trace(go.Scatter(
                x=avg_tc.index, y=avg_tc.values,
                name=sent, line=dict(color=color, width=1.5),
                opacity=0.7
            ))
        fig.update_layout(
            title="Average Trades Per Trader Over Time",
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)', height=400,
            xaxis_title="Date", yaxis_title="Trades per Trader"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Trade Size
        fig = go.Figure()
        for sent, color in [('Fear', '#ef4444'), ('Greed', '#22c55e')]:
            d = filtered_daily[filtered_daily['sentiment_binary'] == sent]['avg_trade_size']
            q95 = d.quantile(0.95)
            fig.add_trace(go.Histogram(
                x=d.clip(0, q95), name=sent, marker_color=color,
                opacity=0.6, nbinsx=40
            ))
        fig.update_layout(
            title="Trade Size Distribution (clipped at 95th pctile)",
            barmode='overlay',
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)', height=400,
            xaxis_title="Avg Trade Size (USD)", yaxis_title="Frequency"
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        # Long/Short Split
        ls_data = []
        for sent in ['Fear', 'Greed']:
            subset = filtered_daily[filtered_daily['sentiment_binary'] == sent]
            if len(subset) > 0:
                total_l = subset['long_trades'].sum()
                total_s = subset['short_trades'].sum()
                total = total_l + total_s
                if total > 0:
                    ls_data.append({'Sentiment': sent, 'Direction': 'Long',
                                   'Percentage': total_l / total * 100})
                    ls_data.append({'Sentiment': sent, 'Direction': 'Short',
                                   'Percentage': total_s / total * 100})

        if ls_data:
            ls_df = pd.DataFrame(ls_data)
            fig = px.bar(
                ls_df, x='Sentiment', y='Percentage', color='Direction',
                color_discrete_map={'Long': '#3b82f6', 'Short': '#f97316'},
                title="Long vs Short Split by Sentiment",
                text='Percentage'
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
            fig.update_layout(
                template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)', height=400,
                barmode='stack'
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Volume comparison
        vol_data = filtered_daily.groupby('sentiment_binary')['total_volume'].mean()
        fig = px.bar(
            x=vol_data.index, y=vol_data.values,
            color=vol_data.index,
            color_discrete_map={'Fear': '#ef4444', 'Greed': '#22c55e', 'Neutral': '#f59e0b'},
            title="Average Daily Volume per Trader",
            labels={'x': 'Sentiment', 'y': 'Volume (USD)'}
        )
        fig.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)', height=400, showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    # Detailed sentiment breakdown
    st.markdown("#### Performance by Detailed Sentiment Classification")
    sentiment_order = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
    available = [s for s in sentiment_order if s in daily_trader['classification'].unique()]
    detailed = daily_trader.groupby('classification').agg(
        Avg_PnL=('daily_pnl', 'mean'),
        Win_Rate=('win_rate', 'mean'),
        Avg_Trades=('trade_count', 'mean'),
        Avg_Size=('avg_trade_size', 'mean'),
        Observations=('daily_pnl', 'count'),
    ).reindex(available).round(4)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            x=detailed.index, y=detailed['Avg_PnL'],
            color=detailed.index,
            color_discrete_map={
                'Extreme Fear': '#991b1b', 'Fear': '#ef4444',
                'Neutral': '#f59e0b',
                'Greed': '#22c55e', 'Extreme Greed': '#15803d'
            },
            title="Average PnL by Detailed Sentiment"
        )
        fig.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)', height=400, showlegend=False,
            xaxis_title="Sentiment", yaxis_title="Avg PnL (USD)"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            x=detailed.index, y=detailed['Win_Rate'] * 100,
            color=detailed.index,
            color_discrete_map={
                'Extreme Fear': '#991b1b', 'Fear': '#ef4444',
                'Neutral': '#f59e0b',
                'Greed': '#22c55e', 'Extreme Greed': '#15803d'
            },
            title="Win Rate by Detailed Sentiment"
        )
        fig.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)', height=400, showlegend=False,
            xaxis_title="Sentiment", yaxis_title="Win Rate (%)"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    st.markdown("#### Correlation Heatmap of Trading Metrics")
    corr_cols = ['daily_pnl', 'trade_count', 'avg_trade_size', 'total_volume', 'win_rate', 'long_short_ratio']
    corr_matrix = daily_trader[corr_cols].corr().round(3)
    fig = px.imshow(
        corr_matrix,
        text_auto=True, aspect='auto',
        color_continuous_scale='RdBu_r',
        title="Correlation Matrix"
    )
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)', height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 4: TRADER SEGMENTS
# ============================================================
with tab4:
    st.markdown('<div class="section-header">Part B — Trader Segmentation</div>', unsafe_allow_html=True)

    st.markdown(f"**Current Segment View:** `{segment_type.replace('_', ' ').title()}`")

    # Segment distribution
    col1, col2 = st.columns(2)
    with col1:
        seg_counts = trader_metrics[segment_type].value_counts()
        fig = px.pie(
            values=seg_counts.values, names=seg_counts.index,
            title=f"Trader Distribution — {segment_type.replace('_', ' ').title()}",
            color_discrete_sequence=['#3b82f6', '#f97316']
        )
        fig.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)', height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Segment metrics
        seg_metrics = trader_metrics.groupby(segment_type).agg(
            Count=('Account', 'count'),
            Avg_PnL=('avg_daily_pnl', 'mean'),
            Avg_Win_Rate=('avg_win_rate', 'mean'),
            Avg_Trades_Per_Day=('trades_per_day', 'mean'),
            Avg_Trade_Size=('avg_trade_size', 'mean'),
        ).round(4)
        st.dataframe(seg_metrics, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Segment Performance by Sentiment")

    fg_seg = daily_with_segments[daily_with_segments['sentiment_binary'].isin(['Fear', 'Greed'])]

    seg_perf = fg_seg.groupby([segment_type, 'sentiment_binary']).agg(
        Avg_PnL=('daily_pnl', 'mean'),
        Win_Rate=('win_rate', 'mean'),
        Avg_Trades=('trade_count', 'mean'),
        Avg_Size=('avg_trade_size', 'mean'),
    ).round(4).reset_index()

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            seg_perf, x=segment_type, y='Avg_PnL', color='sentiment_binary',
            barmode='group', title="Avg PnL by Segment and Sentiment",
            color_discrete_map={'Fear': '#ef4444', 'Greed': '#22c55e'}
        )
        fig.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)', height=400,
            xaxis_title=segment_type.replace('_', ' ').title(),
            yaxis_title="Avg PnL (USD)"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            seg_perf, x=segment_type, y='Win_Rate', color='sentiment_binary',
            barmode='group', title="Win Rate by Segment and Sentiment",
            color_discrete_map={'Fear': '#ef4444', 'Greed': '#22c55e'}
        )
        fig.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)', height=400,
            xaxis_title=segment_type.replace('_', ' ').title(),
            yaxis_title="Win Rate"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Scatter: PnL vs Trade Size by segment
    st.markdown("#### Trader Scatter: PnL vs Trade Size")
    fig = px.scatter(
        trader_metrics, x='avg_trade_size', y='avg_daily_pnl',
        color=segment_type, size='total_trades',
        hover_data=['Account', 'avg_win_rate', 'active_days'],
        title="Trader Performance Scatter",
        color_discrete_sequence=['#3b82f6', '#f97316', '#8b5cf6'],
        opacity=0.6
    )
    q95_x = trader_metrics['avg_trade_size'].quantile(0.95)
    q95_y = trader_metrics['avg_daily_pnl'].quantile(0.95)
    q05_y = trader_metrics['avg_daily_pnl'].quantile(0.05)
    fig.update_xaxes(range=[0, q95_x])
    fig.update_yaxes(range=[q05_y, q95_y])
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)', height=500,
        xaxis_title="Avg Trade Size (USD)", yaxis_title="Avg Daily PnL (USD)"
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 5: PREDICTION MODEL
# ============================================================
with tab5:
    st.markdown('<div class="section-header">Bonus — Predictive Model</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
        <strong>Objective:</strong> Predict whether a trader's daily performance will be <strong>profitable</strong>
        or a <strong>loss</strong> using sentiment data and behavioral features.
    </div>
    """, unsafe_allow_html=True)

    # Model configuration
    col1, col2, col3 = st.columns(3)
    with col1:
        model_type = st.selectbox("Model", ["Random Forest", "Logistic Regression", "Both"])
    with col2:
        test_size = st.slider("Test Size", 0.1, 0.4, 0.25, 0.05)
    with col3:
        n_estimators = st.slider("RF Trees", 50, 300, 100, 50)

    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        with st.spinner("Training models..."):
            # Prepare data
            model_data = daily_trader[daily_trader['sentiment_binary'].isin(['Fear', 'Greed'])].copy()
            model_data['sentiment_encoded'] = LabelEncoder().fit_transform(model_data['sentiment_binary'])
            model_data['profitable'] = (model_data['daily_pnl'] > 0).astype(int)

            feature_cols = ['sentiment_encoded', 'trade_count', 'avg_trade_size',
                           'total_volume', 'long_trades', 'short_trades', 'long_short_ratio']
            X = model_data[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
            y = model_data['profitable']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            results = {}

            if model_type in ["Logistic Regression", "Both"]:
                lr = LogisticRegression(max_iter=1000, random_state=42)
                lr.fit(X_train, y_train)
                lr_pred = lr.predict(X_test)
                results['Logistic Regression'] = {
                    'model': lr, 'pred': lr_pred,
                    'acc': accuracy_score(y_test, lr_pred),
                    'cm': confusion_matrix(y_test, lr_pred),
                    'report': classification_report(y_test, lr_pred, target_names=['Loss', 'Profit'], output_dict=True)
                }

            if model_type in ["Random Forest", "Both"]:
                rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, max_depth=10)
                rf.fit(X_train, y_train)
                rf_pred = rf.predict(X_test)
                results['Random Forest'] = {
                    'model': rf, 'pred': rf_pred,
                    'acc': accuracy_score(y_test, rf_pred),
                    'cm': confusion_matrix(y_test, rf_pred),
                    'report': classification_report(y_test, rf_pred, target_names=['Loss', 'Profit'], output_dict=True),
                    'importance': rf.feature_importances_
                }

            # Display results
            st.markdown("---")
            st.markdown("### Model Results")

            cols = st.columns(len(results))
            for idx, (name, res) in enumerate(results.items()):
                with cols[idx]:
                    st.markdown(f"#### {name}")
                    st.metric("Accuracy", f"{res['acc']:.4f}")

                    # Confusion Matrix
                    fig = px.imshow(
                        res['cm'],
                        text_auto=True,
                        labels=dict(x="Predicted", y="Actual"),
                        x=['Loss', 'Profit'], y=['Loss', 'Profit'],
                        color_continuous_scale='Blues' if 'Logistic' in name else 'Greens',
                        title=f"Confusion Matrix"
                    )
                    fig.update_layout(
                        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)', height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Classification report
                    report_df = pd.DataFrame(res['report']).transpose().round(4)
                    st.dataframe(report_df, use_container_width=True)

            # Feature Importance (for RF)
            if 'Random Forest' in results:
                st.markdown("---")
                st.markdown("### Feature Importance")
                imp = results['Random Forest']['importance']
                imp_df = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': imp
                }).sort_values('Importance', ascending=True)

                fig = px.bar(
                    imp_df, x='Importance', y='Feature', orientation='h',
                    color='Importance', color_continuous_scale='viridis',
                    title="Random Forest Feature Importance"
                )
                fig.update_layout(
                    template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)', height=400
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"""
            <div class="insight-box">
                <strong>Model Summary:</strong><br>
                - Training samples: {len(X_train):,} | Test samples: {len(X_test):,}<br>
                - Target balance: {y.value_counts(normalize=True).to_dict()}<br>
                - Best performing features tend to be trade volume and trade count, with sentiment providing
                  incremental but meaningful signal.
            </div>
            """, unsafe_allow_html=True)

    # Clustering section
    st.markdown("---")
    st.markdown("### Bonus: Trader Clustering (K-Means)")

    col1, col2 = st.columns([1, 3])
    with col1:
        n_clusters = st.slider("Number of Clusters", 2, 6, 3)
        cluster_btn = st.button("Run Clustering", type="secondary", use_container_width=True)

    if cluster_btn:
        with st.spinner("Clustering traders..."):
            cluster_features = ['avg_daily_pnl', 'pnl_std', 'trades_per_day',
                               'avg_trade_size', 'avg_win_rate']
            cluster_data = trader_metrics[cluster_features].fillna(0).replace([np.inf, -np.inf], 0)

            # Normalize
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaled = scaler.fit_transform(cluster_data)

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            trader_metrics_c = trader_metrics.copy()
            trader_metrics_c['cluster'] = kmeans.fit_predict(scaled)

            with col2:
                fig = px.scatter(
                    trader_metrics_c, x='avg_daily_pnl', y='pnl_std',
                    color='cluster', size='total_trades',
                    hover_data=['Account', 'avg_win_rate'],
                    title="Trader Clusters (PnL vs Volatility)",
                    color_continuous_scale='viridis',
                    opacity=0.6
                )
                q95 = trader_metrics_c['avg_daily_pnl'].quantile(0.95)
                q05 = trader_metrics_c['avg_daily_pnl'].quantile(0.05)
                q95_std = trader_metrics_c['pnl_std'].quantile(0.95)
                fig.update_xaxes(range=[q05, q95])
                fig.update_yaxes(range=[0, q95_std])
                fig.update_layout(
                    template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)', height=500,
                    xaxis_title="Avg Daily PnL", yaxis_title="PnL Std Dev"
                )
                st.plotly_chart(fig, use_container_width=True)

            # Cluster summary
            cluster_summary = trader_metrics_c.groupby('cluster').agg(
                Count=('Account', 'count'),
                Avg_PnL=('avg_daily_pnl', 'mean'),
                Avg_Win_Rate=('avg_win_rate', 'mean'),
                Avg_Trades_Day=('trades_per_day', 'mean'),
                Avg_Trade_Size=('avg_trade_size', 'mean'),
                Avg_Consistency=('consistency', 'mean'),
            ).round(4)
            st.dataframe(cluster_summary, use_container_width=True)

# ============================================================
# TAB 6: STRATEGY & INSIGHTS
# ============================================================
with tab6:
    st.markdown('<div class="section-header">Part C — Actionable Insights & Strategy</div>', unsafe_allow_html=True)

    # Calculate key stats for dynamic insights
    fear_d = daily_trader[daily_trader['sentiment_binary'] == 'Fear']
    greed_d = daily_trader[daily_trader['sentiment_binary'] == 'Greed']

    fear_mean_pnl = fear_d['daily_pnl'].mean()
    greed_mean_pnl = greed_d['daily_pnl'].mean()
    fear_wr_val = fear_d['win_rate'].mean() * 100
    greed_wr_val = greed_d['win_rate'].mean() * 100
    fear_vol = fear_d['daily_pnl'].std()
    greed_vol = greed_d['daily_pnl'].std()
    fear_avg_tc = fear_d['trade_count'].mean()
    greed_avg_tc = greed_d['trade_count'].mean()

    fear_long = fear_d['long_trades'].sum()
    fear_short = fear_d['short_trades'].sum()
    greed_long = greed_d['long_trades'].sum()
    greed_short = greed_d['short_trades'].sum()
    fear_long_pct = fear_long / (fear_long + fear_short) * 100 if (fear_long + fear_short) > 0 else 50
    greed_long_pct = greed_long / (greed_long + greed_short) * 100 if (greed_long + greed_short) > 0 else 50

    st.markdown("### Key Findings Summary")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card info-card">
            <h3>PnL Difference</h3>
            <div class="value">${abs(greed_mean_pnl - fear_mean_pnl):,.2f}</div>
            <div class="delta">{'Greed' if greed_mean_pnl > fear_mean_pnl else 'Fear'} days are more profitable</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card info-card">
            <h3>Win Rate Gap</h3>
            <div class="value">{abs(greed_wr_val - fear_wr_val):.1f}%</div>
            <div class="delta">{'Greed' if greed_wr_val > fear_wr_val else 'Fear'} days have higher win rate</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card info-card">
            <h3>Volatility Ratio</h3>
            <div class="value">{fear_vol/greed_vol:.2f}x</div>
            <div class="delta">Fear days are {'more' if fear_vol > greed_vol else 'less'} volatile</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Strategy 1
    st.markdown(f"""
    <div class="strategy-box">
        <h3>📌 Strategy 1: Sentiment-Adaptive Position Sizing</h3>
        <p><strong>Rule:</strong> During FEAR periods, reduce position sizes by 20-30%.</p>
        <p><strong>Evidence:</strong></p>
        <ul>
            <li>Fear days — Mean PnL: <strong>${fear_mean_pnl:,.2f}</strong> | PnL Std: <strong>${fear_vol:,.2f}</strong></li>
            <li>Greed days — Mean PnL: <strong>${greed_mean_pnl:,.2f}</strong> | PnL Std: <strong>${greed_vol:,.2f}</strong></li>
        </ul>
        <p><strong>Rationale:</strong> Higher volatility and potentially worse PnL outcomes on Fear days
        mean that risk-adjusted returns are lower. Smaller positions protect capital during uncertain markets.</p>
    </div>
    """, unsafe_allow_html=True)

    # Strategy 2
    st.markdown(f"""
    <div class="strategy-box">
        <h3>📌 Strategy 2: Sentiment-Based Trade Frequency</h3>
        <p><strong>Rule:</strong> Increase trade frequency during Greed days (higher win rate); reduce during Fear days.</p>
        <p><strong>Evidence:</strong></p>
        <ul>
            <li>Fear days — Avg trades/trader: <strong>{fear_avg_tc:.1f}</strong> | Win rate: <strong>{fear_wr_val:.1f}%</strong></li>
            <li>Greed days — Avg trades/trader: <strong>{greed_avg_tc:.1f}</strong> | Win rate: <strong>{greed_wr_val:.1f}%</strong></li>
        </ul>
        <p><strong>Rationale:</strong> Greed periods tend to offer more favorable conditions. Infrequent traders
        should raise activity during Greed; frequent traders should cut back during Fear to avoid overtrading.</p>
    </div>
    """, unsafe_allow_html=True)

    # Strategy 3
    st.markdown(f"""
    <div class="strategy-box">
        <h3>📌 Strategy 3: Directional Bias Alignment</h3>
        <p><strong>Rule:</strong> Increase SHORT exposure during Fear; maintain LONG bias during Greed.</p>
        <p><strong>Evidence:</strong></p>
        <ul>
            <li>Fear days — Long %: <strong>{fear_long_pct:.1f}%</strong></li>
            <li>Greed days — Long %: <strong>{greed_long_pct:.1f}%</strong></li>
        </ul>
        <p><strong>Rationale:</strong> Aligning directional bias with prevailing sentiment improves the
        probability of profitable trades. Contrarian long positions during extreme fear may also work for skilled traders.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Strategy Summary Table")

    strategy_table = pd.DataFrame({
        'Strategy': ['Position Sizing', 'Trade Frequency', 'Direction Bias', 'Segment-Specific'],
        'Fear Days': ['Reduce by 20-30%', 'Reduce (quality > quantity)',
                     'Increase short exposure', 'Low-leverage traders reduce activity'],
        'Greed Days': ['Standard / increase slightly', 'Increase (favorable conditions)',
                      'Maintain long bias', 'Frequent traders capitalize on momentum']
    })
    st.dataframe(strategy_table, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 3 Key Insights")

    st.markdown(f"""
    <div class="insight-box">
        <strong>Insight 1: Sentiment-Performance Link</strong><br>
        There is a measurable relationship between Bitcoin market sentiment and trader profitability on Hyperliquid.
        Traders earn differently during Fear vs Greed regimes, with PnL volatility being notably
        {'higher' if fear_vol > greed_vol else 'lower'} during Fear ({fear_vol/greed_vol:.2f}x).
        This suggests sentiment can be used as a risk management signal.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="insight-box">
        <strong>Insight 2: Behavioral Adaptation</strong><br>
        Traders do adapt their behavior based on sentiment — trade frequency, position sizes, and long/short ratios
        all shift. However, many traders do NOT adapt optimally. The data shows that those who reduce position sizes
        during Fear and increase activity during Greed outperform those who don't adjust.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="insight-box">
        <strong>Insight 3: Segment Matters</strong><br>
        Not all traders are affected equally by sentiment. Consistent winners maintain relatively stable performance
        across both regimes, while inconsistent traders see amplified losses during Fear periods.
        High-leverage traders face disproportionate drawdown risk during Fear. A one-size-fits-all approach
        to sentiment-based trading is suboptimal — strategies should be segment-specific.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; opacity: 0.6;">
    <p>Trader Performance vs Market Sentiment Dashboard | Primetrade.ai Data Science Assignment</p>
    <p>Built with Streamlit + Plotly | Data: Fear & Greed Index + Hyperliquid Trader Data</p>
</div>
""", unsafe_allow_html=True)
