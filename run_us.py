import yfinance as yf
import pandas as pd
import numpy as np
import requests
import datetime
import os
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings("ignore")

# 讀取 GitHub Secret
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

# ====== 美股設定區 ======
YEARS = 5              # 增加到 5 年數據
TOP_PICK = 5           
MIN_VOLUME = 1500000   # 美股流動性大，門檻調高至 150 萬股
# 重點監控清單
MUST_WATCH = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "AMZN", "QQQ", "SPY", "SOXL"] 

def get_us_stock_list():
    """抓取 S&P 500 成份股清單"""
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        res = requests.get(url, headers=headers, timeout=15)
        df = pd.read_html(res.text)[0]
        # 維基百科的代碼符號處理：將 . 換成 - (符合 yfinance 格式)
        symbols = [str(s).replace('.', '-') for s in df['Symbol'].tolist()]
        # 回傳前 100 檔與必看清單
        return list(set(symbols[:100] + MUST_WATCH))
    except Exception as e:
        print(f"清單抓取失敗: {e}")
        return MUST_WATCH

def compute_features(df):
    """計算美股技術指標"""
    df = df.copy()
    # 1. 動能 (20日/60日)
    df["mom20"] = df["Close"].pct_change(20)
    df["mom60"] = df["Close"].pct_change(60)
    
    # 2. RSI
    delta = df["Close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + up / (down + 1e-9)))
    
    # 3. 量能比 (成交量異常)
    df["vol_ratio"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-9)
    
    # 4. 波動率
    df["volatility"] = df["Close"].pct_change().rolling(20).std()
    
    # 5. 乖離率 (Bias): 判斷是否漲過頭
    df["ma20"] = df["Close"].rolling(20).mean()
    df["bias"] = (df["Close"] - df["ma20"]) / df["ma20"]
    
    return df

def send_to_discord(content):
    if DISCORD_WEBHOOK_URL and content.strip():
        try:
            requests.post(DISCORD_WEBHOOK_URL, json={"content": content}, timeout=15)
        except:
            pass

def run():
    if not DISCORD_WEBHOOK_URL: 
        print("未設定 Webhook URL")
        return
        
    symbols = get_us_stock_list()
    scoring = []
    must_watch_details = []
    # 增加 bias 特徵
    features = ["mom20", "mom60", "rsi", "vol_ratio", "volatility", "bias"]

    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(period=f"{YEARS}y")
            
            if len(df) < 150: continue
                
            df = compute_features(df)
            # 預測目標：未來 5 個交易日的報酬
            df["future_return"] = df["Close"].shift(-5) / df["Close"] - 1
            full_data = df.dropna()
            
            if full_data.empty: continue

            # 訓練模型 (調整參數以適合美股高波動)
            model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
            model.fit(full_data[features], full_data["future_return"])
            
            latest_data = df[features].iloc[-1:]
            pred = model.predict(latest_data)[0]
            
            if sym in MUST_WATCH:
                must_watch_details.append({
                    "sym": sym, "pred": pred, "price": df["Close"].iloc[-1],
                    "sup": df.tail(20)['Low'].min(), "res": df.tail(20)['High'].max()
                })
            
            if df["Volume"].tail(10).mean() >= MIN_VOLUME:
                scoring.append((sym, pred))
        except: 
            continue

    # 1. 發送美股排行榜
    # 取得美國東部時間 (EST)
    est_now = (datetime.datetime.utcnow() - datetime.timedelta(hours=5)).strftime("%Y-%m-%d %H:%M EST")
    
    top_picks = sorted(scoring, key=lambda x: x[1], reverse=True)[:TOP_PICK]
    if top_picks:
        report = f"🇺🇸 **美股 AI 預測排行榜** ({est_now})\n━━━━━━━━━━━━━━━━━━\n"
        for i, (s, p) in enumerate(top_picks):
            emoji = ['🥇','🥈','🥉','📈','📈'][i]
            report += f"{emoji} **{s}**: `+{p:.2%}`\n"
        send_to_discord(report)

    # 2. 發送重點標的細節
    if must_watch_details:
        for item in must_watch_details:
            # 美股預測漲幅超過 2.5% 才給火箭
            status = "🚀" if item['pred'] > 0.025 else ("⚖️" if item['pred'] < -0.02 else "💎")
            msg = f"{status} **{item['sym']}** 深度預測\n"
            msg += f"  - 5日報酬預期: `{item['pred']:+.2%}`\n"
            msg += f"  - 目前價格: `${item['price']:.2f}`\n"
            msg += f"  - 近月支撐/壓力: `${item['sup']:.1f} / ${item['res']:.1f}`"
            send_to_discord(msg)

if __name__ == "__main__":
    run()
