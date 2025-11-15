# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import time
from scanner_12_model import run_scanner
from paper_trader import PaperTrader

st.set_page_config(page_title="12-Model Intraday Scanner", layout="wide")
st.title("12-Model Intraday Scanner (NSE)")

config = st.secrets
trader = PaperTrader()

col1, col2 = st.columns([3, 1])
with col2:
    st.markdown(f"**Time:** {datetime.now().strftime('%I:%M %p IST')}")
    if st.button("Run Scanner"):
        st.cache_data.clear()
    auto = st.checkbox("Auto Refresh (30s)", True)
    if st.button("Start Paper Trading"): trader.start(); st.success("ON")
    if st.button("Stop Paper Trading"): trader.stop(); st.info("OFF")

with col1:
    @st.cache_data(ttl=30)
    def get_signals():
        return run_scanner()
    signals = get_signals() if st.session_state.get("run", True) else pd.DataFrame()

    if not signals.empty:
        signals_display = signals.copy()
        signals_display['Price'] = signals_display['Price'].apply(lambda x: f"₹{x:,.2f}")
        signals_display['Change%'] = signals_display['Change%'].apply(lambda x: f"{x:+.2f}%")
        st.dataframe(signals_display, use_container_width=True)
        fig = go.Figure(go.Bar(x=signals['Symbol'], y=signals['Score'], marker_color='purple'))
        fig.update_layout(title="Signal Score", xaxis_title="Stock", yaxis_title="Score")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No signals. Running scanner...")

st.subheader("Paper Trades")
st.dataframe(trader.get_trades(), use_container_width=True)

if auto:
    time.sleep(30)
    st.rerun()
