# scanner_12_model.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from upstox_client import ApiClient, Configuration, HistoricalDataApi
from pmdarima import auto_arima
from arch import arch_model
import pywt
import xgboost as xgb
from tensorflow.keras.models import load_model
import torch
from hmmlearn import hmm
import os

# --- SECRETS ---
config = st.secrets

# --- UPSTOX ---
conf = Configuration()
conf.api_key = config["upstox"]["api_key"]
conf.api_secret = config["upstox"]["api_secret"]
api_client = ApiClient(conf)
historical_api = HistoricalDataApi(api_client)

# --- MODELS ---
@st.cache_resource
def load_models():
    xgb_model = xgb.Booster()
    xgb_model.load_model("models/xgboost_intraday.pkl")
    cnn_model = load_model("models/cnn_model.h5")
    lstm_model = torch.load("models/lstm_model.pt")
    return xgb_model, cnn_model, lstm_model

xgb_model, cnn_model, lstm_model = load_models()

# --- SYMBOLS ---
with open("nifty500.txt") as f:
    SYMBOLS = [line.strip() for line in f if line.strip()]

# --- HELPERS ---
def fetch_upstox(symbol):
    try:
        end = datetime.now()
        start = end - pd.Timedelta(days=7)
        data = historical_api.get_historical_candle_data(
            symbol, config["scanner"]["interval"], start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        )
        df = pd.DataFrame(data['data']['candles'],
                          columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df = df.astype(float).iloc[::-1]
        return df.tail(int(config["scanner"]["lookback_bars"]))
    except:
        return pd.DataFrame()

def add_features(df):
    close = df['close']
    volume = df['volume']
    df['rsi'] = 100 - (100 / (1 + (close.pct_change().where(lambda x: x > 0, 0).rolling(14).mean() /
                                 close.pct_change().where(lambda x: x < 0, 0).abs().rolling(14).mean())))
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df['bb_upper'] = sma20 + 2 * std20
    df['bb_lower'] = sma20 - 2 * std20
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma20
    df['vol_sma'] = volume.rolling(20).mean()
    df['vol_ratio'] = volume / df['vol_sma']
    df['hurst'] = compute_hurst(close.values)
    return df.dropna()

def compute_hurst(ts):
    lags = range(2, 20)
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] if not np.isnan(poly[0]) else 0.5

# --- MAIN SCANNER ---
def run_scanner():
    results = []
    for symbol in SYMBOLS:
        df = fetch_upstox(symbol)
        if df.empty or len(df) < 100 or df['volume'].iloc[-1] < int(config["scanner"]["min_volume"]):
            continue
        df = add_features(df)
        if df.empty:
            continue

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        score = 0.0

        # 1-4: Fast filters
        if latest['vol_ratio'] < 1.5: continue
        if latest['bb_width'] > df['bb_width'].quantile(0.3): continue
        if not (latest['rsi'] < 30 or latest['rsi'] > 70): continue
        if not (latest['hurst'] > 0.6 or latest['hurst'] < 0.4): continue
        score += 4.7

        # 5-12: Heavy models
        try: arima_model = auto_arima(df['close'][-100:], seasonal=False, max_p=2, max_q=2, suppress_warnings=True)
        except: arima_model = None
        if arima_model:
            pred = arima_model.predict(n_periods=1)[0]
            if abs(pred - latest['close']) / latest['close'] > 0.002: score += 2.0

        try:
            returns = df['close'].pct_change().dropna() * 100
            garch = arch_model(returns[-50:], vol='Garch', p=1, q=1)
            res = garch.fit(disp='off')
            if res.conditional_volatility.iloc[-1] < 0.7 * returns[-20:].std(): score += 2.5
        except: pass

        try:
            coeffs, _ = pywt.cwt(df['close'].values[-50:], scales=np.arange(1, 10), wavelet='mexh')
            if np.std(coeffs[-1]) > df['close'].pct_change().std() * 1.5: score += 1.0
        except: pass

        try:
            features = df[['rsi', 'macd', 'bb_width', 'vol_ratio', 'hurst']].iloc[-1:].values
            pred = xgb_model.predict(xgb.DMatrix(features))[0]
            if pred > 0.6: score += 2.5
        except: pass

        try:
            img = df['close'].values[-50:].reshape(1, 50, 1)
            pred = cnn_model.predict(img, verbose=0)[0][0]
            if pred > 0.7: score += 2.0
        except: pass

        try:
            model = hmm.GaussianHMM(n_components=3, n_iter=100)
            model.fit(df['close'].pct_change().dropna().values.reshape(-1, 1))
            if model.predict(df['close'].pct_change().dropna().values.reshape(-1, 1))[-1] in [0, 2]: score += 1.5
        except: pass

        try:
            seq = torch.tensor(df['close'].values[-20:].reshape(1, 20, 1), dtype=torch.float32)
            with torch.no_grad():
                pred = lstm_model(seq).item()
            if abs(pred - latest['close']) / latest['close'] > 0.003: score += 1.8
        except: pass

        price = latest['close']
        change = (price - prev['close']) / prev['close'] * 100
        results.append({
            'Symbol': symbol.replace(".NS", ""),
            'Price': round(price, 2),
            'Change%': round(change, 2),
            'Score': round(score, 2),
            'Volume': f"{int(latest['volume']):,}"
        })

    if results:
        final = pd.DataFrame(results).sort_values("Score", ascending=False).head(int(config["scanner"]["top_n"]))
        final.to_csv("intraday_signals_12.csv", index=False)
        return final
    return pd.DataFrame()
