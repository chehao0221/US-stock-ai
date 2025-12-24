import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
from xgboost import XGBRegressor
from datetime import datetime, timedelta
import warnings

# =========================
# 基本設定 (美股版)
# =========================
warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE_US = os.path.join(BASE_DIR, "us_history.csv")
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

# =========================
# 支撐/壓力位計算 (美股適用)
# =========================
def calc_support_resistance(df):
    try:
        recent = df.iloc[-20:]
        high, low, close = recent['High'].max(), recent['Low'].min(), recent['Close'].iloc[-1]
        pivot = (high + low + close) / 3
        res = (2 * pivot) - low
        sup = (2 * pivot) - high
        return round(sup, 2), round(res, 2)
    except: return 0, 0

# =========================
# 美股股票池 (S&P 500)
# =========================
def get_us_pool():
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        res = requests.get(url, headers=headers, timeout=10)
        df = pd.read_html(res.text)[0]
        # 維基百科的 '.' 在 yfinance 要換成 '-' (例如 BRK.B -> BRK-B)
        return [s.replace('.', '-') for s in df['Symbol'].tolist()[:500]]
    except: 
        return ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "AMZN", "META"]

# =========================
# 主程序
# =========================
def run_us_market():
    # 1. 準備股票池：七巨頭 + S&P 500
    mag_7 = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "AMZN", "META"]
    pool_stocks = get_us_pool()
    all_watch = list(dict.fromkeys(mag_7 + pool_stocks))
    
    print(f"🇺🇸 開始分析美股市場 {len(all_watch)} 檔標的...")
    
    # 2. 下載數據 (美股建議用 2y 以涵蓋多個循環)
    all_data = yf.download(all_watch, period="2y", auto_adjust=True, group_by="ticker", progress=False)
    mkt_df = yf.download("SPY", period="1y", auto_adjust=True, progress=False)
    
    results = {}
    feats = ["mom20", "bias", "vol_ratio"]
    
    # 3. 逐股 AI 分析
    for s in all_watch:
        try:
            df = all_data[s].dropna()
            if len(df) < 50: continue
            
            # 特徵與標籤
            df["mom20"] = df["Close"].pct_change(20)
            df["bias"] = (df["Close"] - df["Close"].rolling(20).mean()) / df["Close"].rolling(20).mean()
            df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
            df["target"] = df["Close"].shift(-5) / df["Close"] - 1
            
            train = df.dropna().iloc[-300:] # 美股交易日較多，稍增數據量
            model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
            model.fit(train[feats], train["target"])
            
            pred = float(model.predict(df[feats].iloc[-1:])[0])
            sup, res = calc_support_resistance(df)
            
            results[s] = {"p": pred, "c": float(df["Close"].iloc[-1]), "sup": sup, "res": res}
        except: continue

    # 4. 組合報告訊息
    msg = f"🇺🇸 **美股 AI 進階預測報告 ({datetime.now():%Y-%m-%d})**\n"
    msg += "------------------------------------------\n\n"
    
    # --- 區塊一：AI 海選 Top 5 (排除巨頭) ---
    msg += "🏆 **AI 海選 Top 5 (美股潛力股)**\n"
    horses = {k: v for k, v in results.items() if k not in mag_7}
    top_5 = sorted(horses, key=lambda x: horses[x]["p"], reverse=True)[:5]
    
    medals = ["🥇", "🥈", "🥉", "📈", "📈"]
    for i, s in enumerate(top_5):
        r = results[s]
        msg += f"{medals[i]} **{s}**: 預估 `{r['p']:+.2%}`\n"
        msg += f" └ 現價: `{r['c']:.2f}` (支撐: `{r['sup']}` / 壓力: `{r['res']}`)\n"
        
    # --- 區塊二：科技巨頭監控 ---
    msg += "\n💎 **科技巨頭監控 (Magnificent 7)**\n"
    for s in mag_7:
        if s in results:
            r = results[s]
            msg += f"**{s}**: 預估 `{r['p']:+.2%}` | 現價: `{r['c']:.2f}`\n"

    msg += "\n💡 AI 預測僅供參考，美股波動大請注意風險控制。"

    # 5. 發送與儲存歷史
    if WEBHOOK_URL:
        requests.post(WEBHOOK_URL, json={"content": msg[:1900]}, timeout=15)
    else: print(msg)
    
    # 儲存供 5 日後對帳
    new_entries = [{"date": datetime.now().date(), "symbol": s, "pred_p": results[s]['c'], 
                    "pred_ret": results[s]['p'], "settled": "False"} for s in (top_5 + mag_7) if s in results]
    pd.DataFrame(new_entries).to_csv(HISTORY_FILE_US, mode='a', header=not os.path.exists(HISTORY_FILE_US), index=False)

if __name__ == "__main__":
    run_us_market()
