import yfinance as yf
import pandas as pd
import numpy as np
import requests
import datetime
from xgboost import XGBRegressor
import warnings
import os

# 忽略警告訊息
warnings.filterwarnings("ignore")

# 美股專用 Webhook (建議在 GitHub Secrets 設定，若想直接寫死可替換後方網址)
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1452520479825858582/KqzFpKzmuIAhEe2bEIuxb8wSCFY71pxhzkwd89fgQiMh7VjbANCIEm_dX9ZiPeBBJCm9"

YEARS = 3
TOP_PICK = 5
MIN_VOLUME = 1000000  # 美股門檻：日均成交量需 > 100 萬股

# ====== 1. 抓取美股清單 (S&P 500 + 熱門標的) ======
def get_us_list():
    print("🔍 正在獲取美股掃描清單 (S&P 500)...")
    try:
        # 從 Wikipedia 抓取標普 500 成分股
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = table[0]
        sp500 = df['Symbol'].tolist()
        # 加上必看標的 (包含 NVDA, QQQ, SOXX 等)
        must_watch = ["NVDA", "AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "QQQ", "SPY", "SOXX"]
        return list(set(sp500 + must_watch))
    except:
        return ["SPY", "QQQ", "NVDA", "AAPL", "TSLA", "MSFT", "GOOGL", "SOXX"]

# ====== 2. 技術指標計算 ======
def compute_features(df):
    # 動能
    df["mom20"] = df["Close"].pct_change(20)
    df["mom60"] = df["Close"].pct_change(60)
    # 強弱 RSI
    delta = df["Close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + up / (down + 1e-9)))
    # 量價與波動
    df["vol_ratio"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-9)
    df["volatility"] = df["Close"].pct_change().rolling(20).std()
    return df

# ====== 3. 主流程 ======
def run():
    all_symbols = get_us_list()
    # 限制掃描前 300 檔以確保執行速度
    target_symbols = all_symbols[:300]
    print(f"📥 下載美股資料中 (共 {len(target_symbols)} 檔)...")
    
    data = yf.download(target_symbols, period=f"{YEARS}y", progress=False)
    
    scoring = []
    must_watch_list = ["NVDA", "TSLA", "QQQ", "SOXX", "SPY"] # 必看清單
    must_watch_results = []
    features = ["mom20", "mom60", "rsi", "vol_ratio", "volatility"]

    for sym in target_symbols:
        try:
            df = data.xs(sym, axis=1, level=1).dropna(how='all') if len(target_symbols) > 1 else data.dropna(how='all')
            if len(df) < 250: continue
            
            df = compute_features(df)
            df["future_return"] = df["Close"].shift(-5) / df["Close"] - 1
            full_data = df.dropna()
            
            model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.07, random_state=42)
            model.fit(full_data[features], full_data["future_return"])
            
            pred = model.predict(df[features].iloc[-1:])[0]
            
            # 儲存結果
            if sym in must_watch_list:
                must_watch_results.append((sym, pred))
            
            if df["Volume"].tail(20).mean() >= MIN_VOLUME:
                scoring.append((sym, pred))
        except: continue

    # 排序取前五名
    scoring = sorted(scoring, key=lambda x: x[1], reverse=True)[:TOP_PICK]
    
    # 發送 Discord
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    msg = f"🇺🇸 **美股 AI 全市場預測日報** ({today})\n"
    msg += "━━━━━━━━━━━━━━━━━━\n"
    msg += "🏆 **Wall Street Top 5 (未來 5 日看漲)**\n"
    for i, (s, p) in enumerate(scoring):
        medal = ["🥇", "🥈", "🥉", "📈", "📈"][i]
        msg += f"{medal} **{s}**: `+{p:.2%}`\n"
    
    msg += "\n🔍 **美股指標標的追蹤**\n"
    for s, p in must_watch_results:
        icon = "🔥" if p > 0.01 else "💎" if p > 0 else "☁️"
        msg += f"{icon} **{s}**: `+{p:.2%}`\n"
    
    msg += "━━━━━━━━━━━━━━━━━━\n"
    msg += "💡 *註：掃描 S&P 500 成分股。預測結果僅供參考。*"

    requests.post(DISCORD_WEBHOOK_URL, json={"content": msg})
    print("✅ 美股預測結果已發送至 Discord")

if __name__ == "__main__":
    run()