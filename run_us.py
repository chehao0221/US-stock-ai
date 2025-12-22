import yfinance as yf
import pandas as pd
import numpy as np
import requests
import datetime
import os
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings("ignore")

# 讀取環境變數 (確保 GitHub Secret 設為 DISCORD_WEBHOOK_URL)
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

# ====== 設定區 ======
YEARS = 2 
TOP_PICK = 5
MIN_VOLUME = 1000000  # 美股成交量大，設定門檻篩選流動性
# 必看美股清單
MUST_WATCH = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "AMZN", "QQQ", "SPY"] 

def get_us_stock_list():
    """抓取 S&P 500 成份股作為掃描底池"""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        res = requests.get(url, timeout=10)
        df = pd.read_html(res.text)[0]
        # 處理符號轉換，如 BRK.B 轉為 BRK-B 以符合 yfinance 格式
        symbols = [s.replace('.', '-') for s in df['Symbol'].tolist()]
        # 掃描前 80 檔權值股，確保 GitHub Actions 在時限內完成
        return list(set(symbols[:80] + MUST_WATCH))
    except Exception as e:
        print(f"美股清單抓取失敗: {e}")
        return MUST_WATCH

def compute_features(df):
    """計算美股技術指標特徵"""
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
    """分段發送訊息，解決 2000 字元限制導致 400 錯誤的問題"""
    if DISCORD_WEBHOOK_URL and content.strip():
        res = requests.post(DISCORD_WEBHOOK_URL, json={"content": content}, timeout=15)
        print(f"📡 Discord 回傳狀態碼: {res.status_code}")

def run():
    if not DISCORD_WEBHOOK_URL:
        print("❌ 錯誤：未設定 DISCORD_WEBHOOK_URL")
        return

    symbols = get_us_stock_list()
    scoring = []
    must_watch_details = [] 
    features = ["mom20", "mom60", "rsi", "vol_ratio", "volatility"]

    print(f"📡 正在進行美股 AI 掃描 (目標: {len(symbols)} 檔)...")
    
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(period=f"{YEARS}y")
            if len(df) < 120: continue 
            
            df = compute_features(df)
            df["future_return"] = df["Close"].shift(-5) / df["Close"] - 1
            full_data = df.dropna()
            if full_data.empty: continue

            # 訓練 XGBoost 模型
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

    # 1. 發送第一部分：AI 預測排行榜
    today = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    top_picks = sorted(scoring, key=lambda x: x[1], reverse=True)[:TOP_PICK]
    
    header = f"🇺🇸 **美股 AI 掃描報告** ({today})\n━━━━━━━━━━━━━━━━━━\n"
    header += "🏆 **未來 5 日漲幅預測 Top 5**\n"
    for i, (s, p) in enumerate(top_picks):
        header += f"{['🥇','🥈','🥉','📈','📈'][i]} **{s}**: `+{p:.2%}`\n"
    send_to_discord(header)

    # 2. 分段發送第二部分：重點標的深度數據
    for item in must_watch_details:
        status = "🚀" if item['pred'] > 0.01 else "💎"
        detail = f"{status} **{item['sym']}** 數據分析\n"
        detail += f"  - 預測報酬: `{item['pred']:+.2%}`\n"
        detail += f"  - 現價: {item['price']:.2f} (支撐: {item['sup']:.2f} / 壓力: `{item['res']:.2f}`)\n"
        send_to_discord(detail)

if __name__ == "__main__":
    run()
