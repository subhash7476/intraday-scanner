# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import threading
import time
import os

from scanner_12_model import run_scanner
from paper_trader import PaperTrader
from backtest_module import backtest_symbol

st.set_page_config(page_title="12-Model Intraday Scanner", layout="wide")

# --- SIDEBAR ---
st.sidebar.title("12-Model Scanner Dashboard")
st.sidebar.markdown("**NSE Intraday | 5-min | Upstox**")
st.sidebar.markdown(f"**Time:** {datetime.now().strftime('%I:%M %p IST')}")

scan_now = st.sidebar.button("Run Scanner Now")
auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=True)

# --- PAPER TRADER ---
trader = PaperTrader()
if st.sidebar.button("Start Paper Trading"):
    trader.start()
    st.sidebar.success("Paper trading ON")
if st.sidebar.button("Stop Paper Trading"):
    trader.stop()
    st.sidebar.info("Paper trading OFF")

# --- MAIN ---
st.title("Live 12-Model Intraday Signals")
st.markdown("**Cascading Filters | ARIMA + GARCH + DL Ensemble**")

# Cache scanner results
@st.cache_data(ttl=30)
def get_signals():
    return run_scanner()

if scan_now or auto_refresh:
    with st.spinner("Scanning 500 stocks..."):
        signals = get_signals()
        if not signals.empty:
            st.success(f"Found {len(signals)} signals!")
        else:
            signals = pd.DataFrame()
else:
    signals = pd.DataFrame()

# --- LIVE TABLE ---
if not signals.empty:
    st.subheader("Top Signals")
    signals_display = signals.copy()
    signals_display['Price'] = signals_display['Price'].apply(lambda x: f"₹{x:,.2f}")
    signals_display['Change%'] = signals_display['Change%'].apply(lambda x: f"{x:+.2f}%")
    st.dataframe(signals_display, use_container_width=True)

    # Plot
    fig = go.Figure(data=[
        go.Bar(x=signals['Symbol'], y=signals['Score'], marker_color='purple')
    ])
    fig.update_layout(title="Signal Strength", xaxis_title="Stock", yaxis_title="Score")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No signals. Click 'Run Scanner' or wait for auto-refresh.")

# --- PAPER TRADES ---
st.subheader("Paper Trade Log")
trades = trader.get_trades()
if not trades.empty:
    st.dataframe(trades, use_container_width=True)
else:
    st.write("No paper trades yet.")

# --- BACKTEST ---
st.subheader("Backtest Report")
symbol = st.text_input("Symbol for Backtest", "RELIANCE.NS")
if st.button("Run Backtest"):
    with st.spinner("Backtesting..."):
        res = backtest_symbol(symbol, "2025-10-01", "2025-11-15")
        if res:
            st.json(res)
        else:
            st.error("No data")

# Auto-refresh
if auto_refresh:
    time.sleep(30)
    st.rerun()
