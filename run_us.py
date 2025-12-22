import yfinance as yf
import pandas as pd
import numpy as np
import requests
import datetime
from xgboost import XGBRegressor
import warnings
import os

warnings.filterwarnings("ignore")

# 讀取 GitHub Secrets 中的 Webhook
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

def get_us_list():
    try:
        # 抓取 S&P 500 成分股
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        sp500 = table[0]['Symbol'].tolist()
        must_watch = ["NVDA", "TSLA", "QQQ", "SOXX", "SPY", "AAPL"]
        return list(set(sp500 + must_watch))
    except:
        return ["SPY", "QQQ", "NVDA", "AAPL", "TSLA", "SOXX"]

def compute_features(df):
    df["mom20"] = df["Close"].pct_change(20)
    df["mom60"] = df["Close"].pct_change(60)
    delta = df["Close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + up / (down + 1e-9)))
    df["vol_ratio"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-9)
    df["volatility"] = df["Close"].pct_change().rolling(20).std()
    return df

def run():
    symbols = get_us_list()[:300] # 掃描前 300 檔確保效能
    data = yf.download(symbols, period="3y", progress=False)
    
    scoring = []
    must_watch_list = ["NVDA", "TSLA", "QQQ", "SOXX"]
    must_watch_results = []
    features = ["mom20", "mom60", "rsi", "vol_ratio", "volatility"]

    for sym in symbols:
        try:
            df = data.xs(sym, axis=1, level=1).dropna(how='all')
            if len(df) < 250: continue
            df = compute_features(df)
            df["future_return"] = df["Close"].shift(-5) / df["Close"] - 1
            full_data = df.dropna()
            
            model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.07, random_state=42)
            model.fit(full_data[features], full_data["future_return"])
            
            pred = model.predict(df[features].iloc[-1:])[0]
            if sym in must_watch_list: must_watch_results.append((sym, pred))
            if df["Volume"].tail(20).mean() >= 1000000: scoring.append((sym, pred))
        except: continue

    scoring = sorted(scoring, key=lambda x: x[1], reverse=True)[:5]
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    msg = f"🇺🇸 **Wall Street AI Report** ({today})\n"
    msg += "━━━━━━━━━━━━━━━━━━\n"
    msg += "🏆 **Top 5 Picks (5D Forecast)**\n"
    for i, (s, p) in enumerate(scoring):
        msg += f"{['🥇','🥈','🥉','📈','📈'][i]} **{s}**: `+{p:.2%}`\n"
    msg += "\n🔍 **Key Index Watch**\n"
    for s, p in must_watch_results:
        msg += f"📌 **{s}**: `+{p:.2%}`\n"
    msg += "━━━━━━━━━━━━━━━━━━"
    
    requests.post(DISCORD_WEBHOOK_URL, json={"content": msg})

if __name__ == "__main__":
    run()
