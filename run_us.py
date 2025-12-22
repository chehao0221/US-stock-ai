import yfinance as yf
import pandas as pd
import numpy as np
import requests
import datetime
import os
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings("ignore")

# 讀取環境變數
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

# ====== 設定區 ======
YEARS = 2 
TOP_PICK = 5
MIN_VOLUME = 1000000 
MUST_WATCH = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "AMZN", "QQQ", "SPY"] 

def get_us_stock_list():
    """抓取 S&P 500 清單"""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        res = requests.get(url, timeout=10)
        df = pd.read_html(res.text)[0]
        # 轉換符合 yfinance 格式的代碼
        symbols = [s.replace('.', '-') for s in df['Symbol'].tolist()]
        # 掃描前 80 檔權值股，確保執行效率
        return list(set(symbols[:80] + MUST_WATCH))
    except Exception as e:
        print(f"清單抓取失敗: {e}")
        return MUST_WATCH

def compute_features(df):
    df = df.copy()
    df["mom20"] = df["Close"].pct_change(20)
    df["mom60"] = df["Close"].pct_change(60)
    delta = df["Close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + up / (down + 1e-9)))
    df["vol_ratio"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-9)
    df["volatility"] = df["Close"].pct_change().rolling(20).std()
    return df

def send_to_discord(content):
    """分段發送，防止 2000 字元限制報錯"""
    if DISCORD_WEBHOOK_URL and content.strip():
        res = requests.post(DISCORD_WEBHOOK_URL, json={"content": content}, timeout=15)
        print(f"📡 Discord Status: {res.status_code}")

def run():
    if not DISCORD_WEBHOOK_URL:
        print("❌ Error: Webhook URL not found.")
        return

    symbols = get_us_stock_list()
    scoring = []
    must_watch_details = [] 
    features = ["mom20", "mom60", "rsi", "vol_ratio", "volatility"]

    print(f"📡 正在掃描 {len(symbols)} 檔美股標的...")
    
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(period=f"{YEARS}y")
            if len(df) < 120: continue 
            
            df = compute_features(df)
            df["future_return"] = df["Close"].shift(-5) / df["Close"] - 1
            full_data = df.dropna()
            if full_data.empty: continue

            # 使用 XGBoost 進行 5 日報酬預測
            model = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.07, random_state=42)
            model.fit(full_data[features], full_data["future_return"])
            pred = model.predict(df[features].iloc[-1:])[0]
            
            curr_price = df["Close"].iloc[-1]
            hist_20 = df.tail(20)
            res = hist_20['High'].max()
            sup = hist_20['Low'].min()

            if sym in MUST_WATCH:
                must_watch_details.append({"sym": sym, "pred": pred, "price": curr_price, "sup": sup, "res": res})
            if df["Volume"].tail(20).mean() >= MIN_VOLUME:
                scoring.append((sym, pred))
        except: continue

    # 1. 發送排行榜
    today = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    top_picks = sorted(scoring, key=lambda x: x[1], reverse=True)[:TOP_PICK]
    
    header = f"🇺🇸 **美股 AI 分析報告** ({today})\n━━━━━━━━━━━━━━━━━━\n🏆 **未來 5 日漲幅預測 Top 5**\n"
    for i, (s, p) in enumerate(top_picks):
        header += f"{['🥇','🥈','🥉','📈','📈'][i]} **{s}**: `+{p:.2%}`\n"
    send_to_discord(header)

    # 2. 發送重點標的
    for item in must_watch_details:
        status = "🚀" if item['pred'] > 0.01 else "💎"
        detail = f"{status} **{item['sym']}** 數據解析\n"
        detail += f"  - 預測報酬: `{item['pred']:+.2%}`\n"
        detail += f"  - 現價: {item['price']:.2f} (支撐: {item['sup']:.2f} / 壓力: `{item['res']:.2f}`)\n"
        send_to_discord(detail)

if __name__ == "__main__":
    run()
