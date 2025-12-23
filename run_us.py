import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
from xgboost import XGBRegressor
from datetime import datetime, timedelta
import warnings

# 1. 基本設定
warnings.filterwarnings("ignore")
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
HISTORY_FILE = "us_history.csv"

def get_us_300_pool():
    """獲取標普500前300檔股票池"""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        df = pd.read_html(response.text)[0]
        # 修正維基百科符號格式 (例如 BRK.B -> BRK-B)
        symbols = [s.replace('.', '-') for s in df['Symbol'].tolist()[:300]]
        print(f"✅ 成功獲取股票池，共 {len(symbols)} 檔")
        return symbols
    except Exception as e:
        print(f"❌ 獲取清單失敗 ({e})，使用備用大型股清單")
        return ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "AMZN", "META", "AVGO", "COST", "NFLX", "AMD", "SMCI", "BA"]

def compute_features(df):
    """計算技術指標特徵"""
    if df is None or len(df) < 35:
        return None
    df = df.copy()
    try:
        # 計算特徵
        df["mom20"] = df["Close"].pct_change(20)
        df["rsi"] = 100 - (100 / (1 + df["Close"].diff().clip(lower=0).rolling(14).mean() / ((-df["Close"].diff().clip(upper=0)).rolling(14).mean() + 1e-9)))
        df["ma20"] = df["Close"].rolling(20).mean()
        df["bias"] = (df["Close"] - df["ma20"]) / (df["ma20"] + 1e-9)
        df["sup"] = df["Low"].rolling(20).min()
        df["res"] = df["High"].rolling(20).max()
        # 預測目標：未來 5 天報酬
        df["target"] = df["Close"].shift(-5).pct_change(5)
        return df
    except:
        return None

def main():
    # 定義觀察清單與股票池
    must_watch = ["AAPL", "NVDA", "MSFT", "TSLA", "GOOGL"]
    pool = get_us_300_pool()
    all_targets = list(dict.fromkeys(pool + must_watch)) # 去重
    
    results = {}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=200) # 給予足夠歷史資料計算指標
    
    print(f"🚀 開始分析 {len(all_targets)} 檔股票...")
    
    for s in all_targets:
        try:
            # 下載資料
            df = yf.download(s, start=start_date, end=end_date, progress=False, repair=True)
            if df is None or len(df) < 40:
                continue
            
            # 計算指標
            df_feat = compute_features(df)
            if df_feat is None: continue
            
            # 準備訓練模型
            feats = ["mom20", "rsi", "bias"]
            train_data = df_feat.dropna()
            
            if len(train_data) < 15: continue
            
            model = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1)
            model.fit(train_data[feats], train_data["target"])
            
            # 預測
            last_row = df_feat[feats].iloc[-1:]
            pred = model.predict(last_row)[0]
            
            results[s] = {
                "p": float(pred), 
                "c": float(df["Close"].iloc[-1]), 
                "s": float(df_feat["sup"].iloc[-1]), 
                "r": float(df_feat["res"].iloc[-1])
            }
        except Exception as e:
            continue # 遇到報錯跳過該檔

    # --- 排序邏輯 ---
    # 先選出非權值股的 Top 5
    filtered_list = [s for s in results if s not in must_watch]
    top_candidates = sorted(filtered_list, key=lambda x: results[x]['p'], reverse=True)
    
    # 如果非權值股不夠 5 檔，就從權值股裡面補進去
    if len(top_candidates) < 5:
        others = sorted([s for s in results if s in must_watch], key=lambda x: results[x]['p'], reverse=True)
        top_5 = (top_candidates + others)[:5]
    else:
        top_5 = top_candidates[:5]

    # --- 建立訊息 ---
    today_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    msg = f"🇺🇸 **美股 AI 預估報告 ({today_str})**\n"
    msg += "----------------------------------\n"
    
    if not results:
        msg += "❌ 錯誤：無法獲取任何股票資料，請檢查網路或 API。"
    else:
        msg += "🏆 **AI 推薦強勢股 (Top 5)**\n"
        ranks = ["🥇", "🥈", "🥉", "📈", "📈"]
        for idx, s in enumerate(top_5):
            i = results[s]
            msg += f"{ranks[idx]} **{s}**: `預估 {i['p']:+.2%}`\n"
            msg += f"   └ 現價: `${i['c']:.2f}` (支撐: {i['s']:.1f} / 壓力: {i['r']:.1f})\n"

        msg += "\n💡 **大型權值股動態**\n"
        for s in must_watch:
            if s in results:
                i = results[s]
                msg += f"• **{s}**: `{i['p']:+.2%}` (現價: ${i['c']:.2f})\n"

    # 發送 Discord
    if WEBHOOK_URL:
        requests.post(WEBHOOK_URL, json={"content": msg})
    print("✅ 報告已發送至 Discord")

if __name__ == "__main__":
    main()
