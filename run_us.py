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
# 美股歷史記錄檔名
HISTORY_FILE = "us_stock_predictions.csv"

# ====== 美股設定區 ======
YEARS = 5
TOP_PICK = 5
MIN_VOLUME = 1500000 
# 重點監控清單
MUST_WATCH = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "AMZN", "QQQ", "SPY", "SOXL"] 

def get_us_stock_list():
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        res = requests.get(url, headers=headers, timeout=15)
        df = pd.read_html(res.text)[0]
        symbols = [str(s).replace('.', '-') for s in df['Symbol'].tolist()]
        return list(set(symbols[:100] + MUST_WATCH))
    except Exception as e:
        print(f"美股清單抓取失敗: {e}")
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
    df["ma20"] = df["Close"].rolling(20).mean()
    df["bias"] = (df["Close"] - df["ma20"]) / df["ma20"]
    return df

def check_us_accuracy_and_report():
    """自動對帳美股 5 日前的預測"""
    if not os.path.exists(HISTORY_FILE): return ""
    history = pd.read_csv(HISTORY_FILE)
    history['Date'] = pd.to_datetime(history['Date'])
    check_date = datetime.datetime.now() - datetime.timedelta(days=7)
    pending = history[(history['Date'] <= check_date) & (history['Actual_Return'].isna())]
    if pending.empty: return ""

    report = "🇺🇸 **美股 AI 準確度結算 (5日前預測)**\n━━━━━━━━━━━━━━━━━━\n"
    for idx, row in pending.iterrows():
        try:
            ticker = yf.Ticker(row['Symbol'])
            current_price = ticker.history(period="1d")["Close"].iloc[-1]
            actual_ret = (current_price / row['Price_At_Pred']) - 1
            history.at[idx, 'Actual_Return'] = actual_ret
            hit = "🎯" if (actual_ret > 0 and row['Pred_Return'] > 0) or (actual_ret < 0 and row['Pred_Return'] < 0) else "💨"
            report += f"{hit} **{row['Symbol']}**: 預估 `{row['Pred_Return']:+.1%}` / 實際 `{actual_ret:+.1%}`\n"
        except: continue
    history.to_csv(HISTORY_FILE, index=False)
    return report

def save_us_prediction(symbol, pred, price):
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
    
    # 1. 執行美股對帳
    acc_report = check_us_accuracy_and_report()
    if acc_report: requests.post(DISCORD_WEBHOOK_URL, json={"content": acc_report})

    symbols = get_us_stock_list()
    scoring = []
    features = ["mom20", "mom60", "rsi", "vol_ratio", "volatility", "bias"]

    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(period=f"{YEARS}y")
            if len(df) < 150: continue
            
            # 計算美股支撐與壓力 (近 20 日)
            sup = df['Low'].tail(20).min()
            res = df['High'].tail(20).max()
                
            df = compute_features(df)
            df["future_return"] = df["Close"].shift(-5) / df["Close"] - 1
            full_data = df.dropna()
            if full_data.empty: continue

            model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
            model.fit(full_data[features], full_data["future_return"])
            
            latest_price = df["Close"].iloc[-1]
            pred = model.predict(df[features].iloc[-1:])[0]
            
            # 存入海選結果 (符合成交量或在監控清單中)
            if df["Volume"].tail(10).mean() >= MIN_VOLUME or sym in MUST_WATCH:
                scoring.append({
                    "sym": sym, "pred": pred, "price": latest_price,
                    "sup": sup, "res": res
                })
        except: continue

    # 2. 處理美股排行榜報告
    est_now = (datetime.datetime.utcnow() - datetime.timedelta(hours=5)).strftime("%Y-%m-%d %H:%M EST")
    top_picks = sorted(scoring, key=lambda x: x['pred'], reverse=True)[:TOP_PICK]
    
    if top_picks:
        report = f"🇺🇸 **美股 AI 最新預測排行榜** ({est_now})\n━━━━━━━━━━━━━━━━━━\n"
        for i, item in enumerate(top_picks):
            save_us_prediction(item['sym'], item['pred'], item['price'])
            emoji = ['🥇','🥈','🥉','📈','📈'][i]
            report += f"{emoji} **{item['sym']}**: `+{item['pred']:.2%}`\n"
            report += f"   └ 現價: `${item['price']:.2f}` (支撐: {item['sup']:.1f} / 壓力: {item['res']:.1f})\n"
        requests.post(DISCORD_WEBHOOK_URL, json={"content": report})

if __name__ == "__main__":
    run()
