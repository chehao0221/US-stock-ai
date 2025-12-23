import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
from xgboost import XGBRegressor
from datetime import datetime, timedelta
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

# 配置環境變數
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
HISTORY_FILE = "us_history.csv"

def get_us_300_pool():
    """獲取標普500前300檔股票池"""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        # 模擬瀏覽器標頭，防止被維基百科拒絕
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        df = pd.read_html(response.text)[0]
        symbols = [s.replace('.', '-') for s in df['Symbol'].tolist()[:300]]
        print(f"成功獲取股票池，共 {len(symbols)} 檔股票")
        return symbols
    except Exception as e:
        print(f"獲取清單失敗 ({e})，使用備用清單")
        return ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "AMZN", "META", "AVGO", "COST", "NFLX"]

def compute_features(df):
    """計算技術指標特徵"""
    df = df.copy()
    if len(df) < 30:
        return None
    
    # 基礎指標
    df["mom20"] = df["Close"].pct_change(20)
    df["rsi"] = 100 - (100 / (1 + df["Close"].diff().clip(lower=0).rolling(14).mean() / ((-df["Close"].diff().clip(upper=0)).rolling(14).mean() + 1e-9)))
    df["ma20"] = df["Close"].rolling(20).mean()
    df["bias"] = (df["Close"] - df["ma20"]) / (df["ma20"] + 1e-9)
    
    # 支撐壓力位 (簡易版)
    df["sup"] = df["Low"].rolling(20).min()
    df["res"] = df["High"].rolling(20).max()
    
    # 預測目標：未來 5 天的報酬率
    df["target"] = df["Close"].shift(-5).pct_change(5)
    return df

def audit_and_save(results, top_5):
    """保存預測結果到 CSV"""
    new_records = []
    today_str = datetime.now().strftime("%Y-%m-%d")
    for s in top_5:
        if s in results:
            new_records.append({
                "date": today_str,
                "symbol": s,
                "pred_p": results[s]['p'],
                "pred_ret": results[s]['p'], # 預測報酬
                "settled": 0
            })
    
    if new_records:
        new_df = pd.DataFrame(new_records)
        if os.path.exists(HISTORY_FILE):
            old_df = pd.read_csv(HISTORY_FILE)
            pd.concat([old_df, new_df]).to_csv(HISTORY_FILE, index=False)
        else:
            new_df.to_csv(HISTORY_FILE, index=False)
    return True

def main():
    pool = get_us_300_pool()
    must_watch = ["AAPL", "NVDA", "MSFT", "TSLA"]
    results = {}
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=150)
    
    print("開始下載資料與模型預測...")
    for s in pool + must_watch:
        try:
            # 下載資料，加入 repair=True 增加成功率
            df = yf.download(s, start=start_date, end=end_date, progress=False, repair=True)
            
            if df is None or len(df) < 50:
                continue
                
            df = compute_features(df)
            if df is None: continue
            
            feats = ["mom20", "rsi", "bias"]
            train = df.dropna()
            
            if len(train) < 10: continue
            
            # XGBoost 模型
            model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.07)
            model.fit(train[feats], train["target"])
            
            # 進行最後一天的預測
            last_row = df[feats].iloc[-1:]
            pred = model.predict(last_row)[0]
            
            results[s] = {
                "p": float(pred), 
                "c": float(df["Close"].iloc[-1]), 
                "s": float(df["sup"].iloc[-1]), 
                "r": float(df["res"].iloc[-1])
            }
        except Exception as e:
            print(f"處理 {s} 時出錯: {e}")
            continue

    # 排序：排除權值股後的 Top 5
    filtered_list = [s for s in results if s not in must_watch]
    top_5 = sorted(filtered_list, key=lambda x: results[x]['p'], reverse=True)[:5]
    
    # 存檔
    audit_and_save(results, top_5)
    
    # 建立 Discord 訊息
    today = datetime.now().strftime("%Y-%m-%d %H:%M EST")
    msg = f"🇺🇸 **美股 AI 預估報告 ({today})**\n"
    msg += "----------------------------------\n"
    
    if not top_5:
        msg += "⚠️ 今日無符合條件之推薦股票。\n"
    else:
        msg += "🏆 **300 股票前 5 的未來預估**\n"
        ranks = ["🥇", "🥈", "🥉", "📈", "📈"]
        for idx, s in enumerate(top_5):
            i = results[s]
            msg += f"{ranks[idx]} **{s}**: `預估 {i['p']:+.2%}`\n"
            msg += f"   └ 現價: `${i['c']:.2f}` (支撐: {i['s']:.1f} / 壓力: {i['r']:.1f})\n"

    msg += "\n💡 **權值股觀察**\n"
    for s in must_watch:
        if s in results:
            i = results[s]
            msg += f"• **{s}**: `{i['p']:+.2%}` (現價: {i['c']:.2f})\n"

    # 發送 Discord Webhook
    if WEBHOOK_URL:
        requests.post(WEBHOOK_URL, json={"content": msg})
    else:
        print("未設定 Webhook URL，僅輸出結果：")
        print(msg)

if __name__ == "__main__":
    main()
