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
# 美股專用歷史紀錄檔名
HISTORY_FILE = "us_stock_predictions.csv"

# ====== 美股設定區 ======
YEARS = 5
TOP_PICK = 5
MIN_VOLUME = 1500000 
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
    """自動對帳美股約 5 個交易日前預測的準確度"""
    if not os.path.exists(HISTORY_FILE): return ""
    
    history = pd.read_csv(HISTORY_FILE)
    history['Date'] = pd.to_datetime(history['Date'])
    # 回溯約 7 天（包含週末）來檢查 5 天前的預測
    check_date = datetime.datetime.now() - datetime.timedelta(days=7)
    
    # 篩選出日期已到且尚未對帳的紀錄
    pending = history[(history['Date'] <= check_date) & (history['Actual_Return'].isna())]
    if pending.empty: return ""

    report = "🇺🇸 **美股 AI 預測結算報告 (5日前預測對帳)**\n━━━━━━━━━━━━━━━━━━\n"
    for idx, row in pending.iterrows():
        try:
            ticker = yf.Ticker(row['Symbol'])
            # 取得最新收盤價
            current_price = ticker.history(period="1d")["Close"].iloc[-1]
            actual_ret = (current_price / row['Price_At_Pred']) - 1
            
            # 更新 dataframe 紀錄
            history.at[idx, 'Actual_Return'] = actual_ret
            
            # 判斷 AI 猜測方向是否正確
            hit = "🎯" if (actual_ret > 0 and row['Pred_Return'] > 0) or (actual_ret < 0 and row['Pred_Return'] < 0) else "💨"
            report += f"{hit} **{row['Symbol']}**: 預估 `{row['Pred_Return']:+.1%}` / 實際 `{actual_ret:+.1%}`\n"
        except Exception as e:
            continue
    
    # 儲存更新後的紀錄
    history.to_csv(HISTORY_FILE, index=False)
    return report

def save_us_prediction(symbol, pred, price):
    """存檔美股當日預測結果"""
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    new_data = pd.DataFrame([[date, symbol, price, pred, np.nan]], 
                            columns=["Date", "Symbol", "Price_At_Pred", "Pred_Return", "Actual_Return"])
    
    if os.path.exists(HISTORY_FILE):
        history = pd.read_csv(HISTORY_FILE)
        history = pd.concat([history, new_data], ignore_index=True)
    else:
        history = new_data
    # 只保留最後 1000 筆紀錄
    history.tail(1000).to_csv(HISTORY_FILE, index=False)

def send_to_discord(content):
    if DISCORD_WEBHOOK_URL and content.strip():
        try:
            requests.post(DISCORD_WEBHOOK_URL, json={"content": content}, timeout=15)
        except: pass

def run():
    if not DISCORD_WEBHOOK_URL: return
    
    # 1. 執行美股準確度對帳並發送報告
    acc_report = check_us_accuracy_and_report()
    if acc_report: send_to_discord(acc_report)

    symbols = get_us_stock_list()
    scoring = []; must_watch_details = []
    features = ["mom20", "mom60", "rsi", "vol_ratio", "volatility", "bias"]

    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(period=f"{YEARS}y")
            if len(df) < 150: continue
                
            df = compute_features(df)
            df["future_return"] = df["Close"].shift(-5) / df["Close"] - 1
            full_data = df.dropna()
            if full_data.empty: continue

            model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
            model.fit(full_data[features], full_data["future_return"])
            
            latest_price = df["Close"].iloc[-1]
            pred = model.predict(df[features].iloc[-1:])[0]
            
            if sym in MUST_WATCH:
                must_watch_details.append({
                    "sym": sym, "pred": pred, "price": latest_price,
                    "sup": df.tail(20)['Low'].min(), "res": df.tail(20)['High'].max()
                })
            
            if df["Volume"].tail(10).mean() >= MIN_VOLUME:
                scoring.append((sym, pred, latest_price))
        except: continue

    # 2. 處理最新預測排行榜
    est_now = (datetime.datetime.utcnow() - datetime.timedelta(hours=5)).strftime("%Y-%m-%d %H:%M EST")
    top_picks = sorted(scoring, key=lambda x: x[1], reverse=True)[:TOP_PICK]
    
    if top_picks:
        report = f"🇺🇸 **美股 AI 最新預測排行榜** ({est_now})\n━━━━━━━━━━━━━━━━━━\n"
        for i, (s, p, price) in enumerate(top_picks):
            save_us_prediction(s, p, price) # 存入美股歷史紀錄
            emoji = ['🥇','🥈','🥉','📈','📈'][i]
            report += f"{emoji} **{s}**: `+{p:.2%}` (現價: `${price:.2f}`)\n"
        send_to_discord(report)

    # 3. 重點監控標的細節
    for item in must_watch_details:
        status = "🚀" if item['pred'] > 0.025 else ("⚖️" if item['pred'] < -0.02 else "💎")
        msg = f"{status} **{item['sym']}**預測: `{item['pred']:+.2%}` | 價格 `${item['price']:.2f}`"
        send_to_discord(msg)

if __name__ == "__main__":
    run()
