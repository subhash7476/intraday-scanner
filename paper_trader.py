# paper_trader.py
import threading
import time
import pandas as pd
from datetime import datetime
import json
from telegram import Bot

with open("config.json") as f:
    config = json.load(f)

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
                df = pd.read_csv("intraday_signals_12.csv")
                if df.empty:
                    time.sleep(15)
                    continue

                top = df.iloc[0]
                symbol = top['Symbol'] + ".NS"
                price = top['Price']
                score = top['Score']

                if score > 14:  # High conviction
                    self._execute_paper_trade(symbol, price, "BUY")
                    time.sleep(300)  # Hold 5 min
                    self._execute_paper_trade(symbol, price * 1.003, "SELL")
            except:
                pass
            time.sleep(15)

    def _execute_paper_trade(self, symbol, price, action):
        qty = 10
        new_trade = {
            'Time': datetime.now().strftime("%H:%M:%S"),
            'Symbol': symbol.replace(".NS", ""),
            'Action': action,
            'Price': round(price, 2),
            'Qty': qty,
            'PnL': 0 if action == "BUY" else round((price - self.trades.iloc[-1]['Price']) * qty, 2) if len(self.trades) > 0 and self.trades.iloc[-1]['Action'] == "BUY" else 0
        }
        self.trades = pd.concat([self.trades, pd.DataFrame([new_trade])], ignore_index=True)
        self.trades.to_csv("paper_trades.csv", index=False)

        msg = f"<b>Paper Trade</b>\n"
        msg += f"• {action} {symbol.replace('.NS', '')}\n"
        msg += f"• Price: ₹{price:,.2f}\n"
        msg += f"• Time: {datetime.now().strftime('%I:%M %p')}"
        bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode='HTML')

    def get_trades(self):
        if os.path.exists("paper_trades.csv"):
            return pd.read_csv("paper_trades.csv")
        return pd.DataFrame()
