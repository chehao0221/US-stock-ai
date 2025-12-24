import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
import warnings
from xgboost import XGBRegressor
from datetime import datetime, timedelta

# =========================
# 基本設定
# =========================
warnings.filterwarnings("ignore")
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
HISTORY_FILE = "us_sp500_history.csv"

def get_sp500_300_pool():
    """從維基百科抓取 S&P 500 清單並取前 300 檔"""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=15)
        df = pd.read_html(res.text)[0]
        # 美股代碼轉換 (例如 BRK.B 轉 BRK-B)
        symbols = [s.replace('.', '-') for s in df['Symbol'].tolist()]
        return symbols[:300]
    except Exception as e:
        print(f"獲取 S&P 500 清單失敗: {e}")
        # 備用核心權值股
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO"]

def compute_features(df):
    """計算美股特徵指標"""
    df = df.copy()
    # 價格變動與動能
    df["mom20"] = df["Close"].pct_change(20)
    
    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / (loss + 1e-9)))
    
    # 乖離率與量能比
    df["ma20"] = df["Close"].rolling(20).mean()
    df["bias"] = (df["Close"] - df["ma20"]) / (df["ma20"] + 1e-9)
    df["vol_ratio"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-9)
    
    # 支撐壓力位
    df["sup"] = df["Low"].rolling(60).min()
    df["res"] = df["High"].rolling(60).max()
    return df

def audit_and_save(results, top_keys):
    """對帳與存檔邏輯"""
    audit_msg = ""
    today = datetime.now().date()
    
    if os.path.exists(HISTORY_FILE):
        hist = pd.read_csv(HISTORY_FILE)
        hist['date'] = pd.to_datetime(hist['date']).dt.date
        
        # 檢查 7 天前的預測
        deadline = today - timedelta(days=7)
        unsettled = hist[(hist['settled'] == False) & (hist['date'] <= deadline)]
        
        if not unsettled.empty:
            audit_msg = "\n🎯 **US 5-Day Prediction Audit**\n"
            for idx, r in unsettled.iterrows():
                try:
                    p_df = yf.Ticker(r["symbol"]).history(period="1d")
                    if p_df.empty: continue
                    curr_p = p_df["Close"].iloc[-1]
                    act_ret = (curr_p - r["pred_p"]) / r["pred_p"]
                    hit = "✅" if np.sign(act_ret) == np.sign(r["pred_ret"]) else "❌"
                    audit_msg += f"`{r['symbol']}`: {r['pred_ret']:+.2%} ➔ {act_ret:+.2%} {hit}\n"
                    hist.at[idx, "settled"] = True
                except: continue
        hist.to_csv(HISTORY_FILE, index=False)
    else:
        hist = pd.DataFrame(columns=["date", "symbol", "pred_p", "pred_ret", "settled"])

    # 存入今日預測
    new_rows = [{"date": today, "symbol": s, "pred_p": results[s]["c"], "pred_ret": results[s]["p"], "settled": False} for s in top_keys]
    hist = pd.concat([hist, pd.DataFrame(new_rows)], ignore_index=True)
    hist.to_csv(HISTORY_FILE, index=False)
    return audit_msg

def run():
    print("🚀 啟動美股 S&P 300 AI 掃描...")
    watch_pool = get_sp500_300_pool()
    must_watch = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL"]
    all_syms = list(set(watch_pool + must_watch))
    
    # 批量抓取資料 (美股建議抓 2 年即可滿足指標計算)
    data = yf.download(all_syms, period="2y", progress=False, group_by="ticker")
    
    results = {}
    feats = ["mom20", "rsi", "bias", "vol_ratio"]
    
    for s in all_syms:
        try:
            df = data[s].dropna()
            if len(df) < 80: continue
            
            df = compute_features(df)
            df["target"] = df["Close"].shift(-5) / df["Close"] - 1 # 預估 5 日後
            
            train = df.dropna()
            model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
            model.fit(train[feats], train["target"])
            
            # 獲取最新一筆進行預測
            pred = float(np.clip(model.predict(df[feats].iloc[-1:])[0], -0.2, 0.2))
            results[s] = {
                "p": pred, 
                "c": df["Close"].iloc[-1],
                "s": df["sup"].iloc[-1],
                "r": df["res"].iloc[-1]
            }
        except: continue

    # 排序前 5 名 (排除必看標的，尋找潛力股)
    top_5 = sorted([s for s in results if s not in must_watch], key=lambda x: results[x]['p'], reverse=True)[:5]
    audit_report = audit_and_save(results, top_5)
    
    # Discord 報告排版
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    msg = f"🇺🇸 **美股 AI 預估報告 (S&P 300) - {report_date}**\n"
    msg += "━━━━━━━━━━━━━━━━━━\n"
    msg += "🏆 **未來 5 日漲幅 Top 5 潛力股**\n"
    
    ranks = ["🥇", "🥈", "🥉", "📈", "📈"]
    for idx, s in enumerate(top_5):
        i = results[s]
        msg += f"{ranks[idx]} **{s}**: `預估 {i['p']:+.2%}`\n"
        msg += f"   └ 現價: `${i['c']:.2f}` (支撐: {i['s']:.2f} / 壓力: {i['r']:.2f})\n"

    msg += "\n💡 **核心權值股觀測**\n"
    for s in must_watch:
        if s in results:
            i = results[s]
            msg += f"⭐ **{s}**: `${i['c']:.2f}` | `預估 {i['p']:+.2%}`\n"

    msg += audit_report + "\n*Risk Warning: Predictions are for educational purposes.*"
    
    # 發送通知
    if WEBHOOK_URL:
        requests.post(WEBHOOK_URL, json={"content": msg})
    else:
        print("\n--- Discord Preview ---\n", msg)

if __name__ == "__main__":
    run()
