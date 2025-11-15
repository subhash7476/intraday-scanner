# paper_trader.py
import threading
import time
import pandas as pd
from datetime import datetime
import streamlit as st
from telegram import Bot
import os

config = st.secrets
bot = Bot(token=config["telegram"]["bot_token"])
CHAT_ID = config["telegram"]["chat_id"]

class PaperTrader:
    def __init__(self):
        self.trades = pd.DataFrame(columns=['Time', 'Symbol', 'Action', 'Price', 'Qty', 'PnL'])
        self.running = False
        self.thread = None

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False

    def _run(self):
        while self.running:
            try:
                if os.path.exists("intraday_signals_12.csv"):
                    df = pd.read_csv("intraday_signals_12.csv")
                    if not df.empty and df.iloc[0]['Score'] > 14:
                        top = df.iloc[0]
                        self._trade(top['Symbol'] + ".NS", top['Price'], "BUY")
                        time.sleep(300)
                        self._trade(top['Symbol'] + ".NS", top['Price'] * 1.003, "SELL")
            except: pass
            time.sleep(15)

    def _trade(self, symbol, price, action):
        qty = 10
        new = {'Time': datetime.now().strftime("%H:%M"), 'Symbol': symbol.replace(".NS",""), 'Action': action, 'Price': round(price,2), 'Qty': qty, 'PnL': 0}
        if action == "SELL" and len(self.trades) > 0 and self.trades.iloc[-1]['Action'] == "BUY":
            new['PnL'] = round((price - self.trades.iloc[-1]['Price']) * qty, 2)
        self.trades = pd.concat([self.trades, pd.DataFrame([new])], ignore_index=True)
        self.trades.to_csv("paper_trades.csv", index=False)
        bot.send_message(chat_id=CHAT_ID, text=f"Paper {action} {symbol.replace('.NS','')} @ ₹{price:,.2f}", parse_mode='HTML')

    def get_trades(self):
        return pd.read_csv("paper_trades.csv") if os.path.exists("paper_trades.csv") else pd.DataFrame()
