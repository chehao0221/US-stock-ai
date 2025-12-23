import yfinance as yf
import pandas as pd
import requests
import os
from xgboost import XGBRegressor
from datetime import datetime

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

def run():
    stocks = ["AAPL", "NVDA", "TSLA"]
    for sym in stocks:
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(period="5y")
            # 簡化特徵計算
            df["mom20"] = df["Close"].pct_change(20)
            df["future_return"] = df["Close"].shift(-5) / df["Close"] - 1
            train = df.dropna()
            
            model = XGBRegressor(n_estimators=50)
            model.fit(train[["mom20"]], train["future_return"])
            
            pred = model.predict(df[["mom20"]].iloc[-1:])[0]
            price = df['Close'].iloc[-1]
            change = ((price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
            trend = "▲" if change > 0 else "▼" if change < 0 else "—"

            payload = {
                "embeds": [{
                    "title": f"🇺🇸 美股 AI 預測: {sym}",
                    "description": f"**現價:** `${price:.2f}` ({trend} `{change:+.2f}%`)\n**AI 5日預估:** `{pred:+.2%}`",
                    "color": 0x36393f # 中性灰色
                }]
            }
            requests.post(WEBHOOK_URL, json=payload)
        except: pass

if __name__ == "__main__":
    run()
