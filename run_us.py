import yfinance as yf
import pandas as pd
import numpy as np
import requests
import datetime
import os
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings("ignore")

# 從 GitHub Secrets 獲取 Webhook 網址
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
HISTORY_FILE = "us_stock_predictions.csv"

# ====== 美股設定區 ======
YEARS = 5
TOP_PICK = 5
MIN_VOLUME = 1500000 
MUST_WATCH = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "AMZN", "QQQ", "SPY", "SOXL"] 

def get_us_stock_list():
    """抓取 S&P 500 前 300 支股票作為選股池"""
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        res = requests.get(url, headers=headers, timeout=15)
        df = pd.read_html(res.text)[0]
        symbols = [str(s).replace('.', '-') for s in df['Symbol'].tolist()]
        return list(set(symbols[:300] + MUST_WATCH))
    except:
        return MUST_WATCH

def compute_features(df):
    """AI 特徵工程：計算技術指標"""
    df = df.copy()
    df["mom20"] = df["Close"].pct_change(20)
    df["mom60"] = df["Close"].pct_change(60)
    delta = df["Close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + up / (down + 1e-9)))
    df["vol_ratio"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-9)
    df["volatility"] = df["Close"].pct_change().rolling(20).std()
    df["ma20"] = df["Close"].rolling(20).mean()
    df["bias"] = (df["Close"] - df["ma20"]) / df["ma20"]
    return df

def send_embed(title, description, color=0x00FF00, fields=None):
    """輔助函式：發送美化的 Discord Embed"""
    payload = {
        "embeds": [{
            "title": title,
            "description": description,
            "color": color,
            "fields": fields if fields else [],
            "footer": {"text": f"美東時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"}
        }]
    }
    requests.post(DISCORD_WEBHOOK_URL, json=payload)

def check_us_accuracy_and_report():
    """回測 7 天前的預測是否精準"""
    if not os.path.exists(HISTORY_FILE): return
    history = pd.read_csv(HISTORY_FILE)
    history['Date'] = pd.to_datetime(history['Date'])
    
    # 檢查 7 天前的資料
    check_date = datetime.datetime.now() - datetime.timedelta(days=7)
    pending = history[(history['Date'].dt.date <= check_date.date()) & (history['Actual_Return'].isna())]
    
    if pending.empty: return

    fields = []
    for idx, row in pending.iterrows():
        try:
            ticker = yf.Ticker(row['Symbol'])
            current_price = ticker.history(period="1d")["Close"].iloc[-1]
            actual_ret = (current_price / row['Price_At_Pred']) - 1
            history.at[idx, 'Actual_Return'] = actual_ret
            
            hit = "🎯" if (actual_ret * row['Pred_Return'] > 0) else "💨"
            fields.append({
                "name": f"{hit} {row['Symbol']}",
                "value": f"預估 `{row['Pred_Return']:+.1%}` / 實際 `{actual_ret:+.1%}`",
                "inline": True
            })
        except: continue
    
    history.to_csv(HISTORY_FILE, index=False)
    if fields:
        send_embed("🇺🇸 美股 AI 預測準確度結算 (5日前預測)", "這份報告回顧上週預測與目前現價的差異。", 0x3498db, fields)

def save_us_prediction(symbol, pred, price):
    """保存預測結果至 CSV 供未來對帳"""
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    new_data = pd.DataFrame([[date, symbol, price, pred, np.nan]], 
                            columns=["Date", "Symbol", "Price_At_Pred", "Pred_Return", "Actual_Return"])
    if os.path.exists(HISTORY_FILE):
        history = pd.read_csv(HISTORY_FILE)
        history = pd.concat([history, new_data], ignore_index=True)
    else: history = new_data
    history.tail(1000).to_csv(HISTORY_FILE, index=False)

def run():
    if not DISCORD_WEBHOOK_URL: return
    
    # 1. 結算舊單
    check_us_accuracy_and_report()

    # 2. 開始 AI 分析
    symbols = get_us_stock_list()
    all_results = {}
    features = ["mom20", "mom60", "rsi", "vol_ratio", "volatility", "bias"]

    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(period=f"{YEARS}y")
            if len(df) < 150: continue
            
            sup, res = df['Low'].tail(20).min(), df['High'].tail(20).max()
            df = compute_features(df)
            df["future_return"] = df["Close"].shift(-5) / df["Close"] - 1
            
            full_data = df.dropna()
            if full_data.empty: continue
            
            model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
            model.fit(full_data[features], full_data["future_return"])
            
            latest_price = df["Close"].iloc[-1]
            pred = model.predict(df[features].iloc[-1:])[0]
            all_results[sym] = {"pred": pred, "price": latest_price, "sup": sup, "res": res, "vol": df["Volume"].tail(10).mean()}
        except: continue

    # 3. 排名與發送報告
    ranking_list = [s for s, v in all_results.items() if v['vol'] >= MIN_VOLUME]
    top_picks_keys = sorted(ranking_list, key=lambda x: all_results[x]['pred'], reverse=True)[:TOP_PICK]
    
    # 發送 Top 5 報告 (美股配色 0x00FF00 為綠色/上漲)
    top_fields = []
    for i, sym in enumerate(top_picks_keys):
        item = all_results[sym]
        save_us_prediction(sym, item['pred'], item['price'])
        top_fields.append({
            "name": f"NO.{i+1} {sym}",
            "value": f"預估: `{item['pred']:+.2%}`\n現價: `${item['price']:.2f}`\n(支撐: {item['sup']:.1f} / 壓力: {item['res']:.1f})",
            "inline": False
        })
    send_embed("🏆 美股 AI 強勢選股 Top 5", "基於 300 支 S&P500 權重股分析，預估 5 日後收益率。", 0x00FF00, top_fields)

    # 發送監控標的報告
    watch_fields = []
    for sym in MUST_WATCH:
        if sym in all_results:
            item = all_results[sym]
            color_emoji = "🟢" if item['pred'] > 0 else "🔴"
            watch_fields.append({
                "name": f"{color_emoji} {sym}",
                "value": f"預估: `{item['pred']:+.2%}` | 現價: `${item['price']:.2f}`",
                "inline": True
            })
    send_embed("💎 指定監控標的預測", "包含您最關注的科技巨頭與 ETF 走勢判斷。", 0xf1c40f, watch_fields)

if __name__ == "__main__":
    run()
