import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
from xgboost import XGBRegressor
from datetime import datetime, timedelta
import warnings

# =========================
# 基本設定與環境
# =========================
warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 依執行檔名自動切換紀錄檔
HISTORY_FILE = os.path.join(BASE_DIR, "trading_history.csv")
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

# =========================
# 進階特徵工程 (核心優化)
# =========================
def compute_features(df, market_df=None):
    df = df.copy()
    
    # 1. 基礎動能與超買超賣
    df["mom20"] = df["Close"].pct_change(20)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / (loss + 1e-9)))
    
    # 2. 乖離率與量比
    df["ma20"] = df["Close"].rolling(20).mean()
    df["bias"] = (df["Close"] - df["ma20"]) / (df["ma20"] + 1e-9)
    df["vol_ratio"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-9)
    
    # 3. 進階指標：ATR (波動率調整)
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df["atr"] = ranges.max(axis=1).rolling(14).mean()
    
    # 4. 進階指標：相對強度 (RS) - 與大盤對比
    if market_df is not None:
        # 計算個股與大盤的 20 日報酬率差
        stock_ret = df["Close"].pct_change(20)
        market_ret = market_df["Close"].pct_change(20)
        df["rs_index"] = stock_ret - market_ret
    else:
        df["rs_index"] = 0

    # 5. 成交金額 (流動性)
    df["avg_amount"] = (df["Close"] * df["Volume"]).rolling(5).mean()
    
    # 6. 支撐壓力
    df["sup"] = df["Low"].rolling(60).min()
    df["res"] = df["High"].rolling(60).max()
    
    return df

# =========================
# 市場趨勢濾網
# =========================
def get_market_context(market_ticker="^TWII"):
    try:
        idx = yf.download(market_ticker, period="1y", auto_adjust=True, progress=False)
        if idx.empty: return True, 0, 0, None
        idx["ma60"] = idx["Close"].rolling(60).mean()
        curr_p = float(idx["Close"].iloc[-1])
        ma60_p = float(idx["ma60"].iloc[-1])
        return curr_p > ma60_p, curr_p, ma60_p, idx
    except:
        return True, 0, 0, None

# =========================
# 主流程
# =========================
def run(market_type="TW"):
    # 設定參數
    if market_type == "TW":
        market_ticker = "^TWII"
        min_amount = 100_000_000  # 1億台幣
        must_watch = ["2330.TW", "2317.TW", "2454.TW", "0050.TW"]
        pool_func = get_tw_pool
    else:
        market_ticker = "^GSPC"
        min_amount = 10_000_000   # 1000萬美金
        must_watch = ["AAPL", "NVDA", "TSLA", "MSFT"]
        pool_func = get_us_pool

    # 1. 取得大盤數據
    is_bull, mkt_p, mkt_ma, mkt_df = get_market_context(market_ticker)
    
    # 2. 取得選股池
    watch = list(set(must_watch + pool_func()))
    
    # 3. 下載數據並處理
    print(f"[{market_type}] 正在分析 {len(watch)} 檔標的...")
    all_data = yf.download(watch, period="5y", group_by="ticker", auto_adjust=True, progress=False)
    
    feats = ["mom20", "rsi", "bias", "vol_ratio", "rs_index"]
    results = {}

    for s in watch:
        try:
            df = all_data[s].dropna()
            if len(df) < 120: continue
            
            df = compute_features(df, market_df=mkt_df)
            last = df.iloc[-1]
            
            # 硬性過濾：流動性不足則跳過
            if last["avg_amount"] < min_amount: continue

            # 訓練模型
            df["target"] = df["Close"].shift(-5) / df["Close"] - 1
            train = df.dropna()
            
            model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
            model.fit(train[feats], train["target"])
            
            # 預測
            pred = float(np.clip(model.predict(train[feats].iloc[-1:])[0], -0.15, 0.15))
            
            # --- 風險干預邏輯 ---
            # 1. 大盤空頭降權
            if not is_bull: pred *= 0.5
            # 2. 波動率過高 (ATR) 降權 - 預防暴漲暴跌
            if last["atr"] > (df["atr"].mean() * 1.5): pred *= 0.8

            results[s] = {
                "p": pred, "c": float(last["Close"]), 
                "amt": float(last["avg_amount"]), "rs": float(last["rs_index"])
            }
        except: continue

    # 4. 產出報告
    horses = {k: v for k, v in results.items() if k not in must_watch}
    top_keys = sorted(horses, key=lambda x: horses[x]['p'], reverse=True)[:5]
    
    # 格式化訊息
    msg = f"{'🇹🇼 台股' if market_type=='TW' else '🇺🇸 美股'} AI 進階預報 ({datetime.now():%m/%d})\n"
    msg += f"{'📈 多頭環境' if is_bull else '⚠️ 空頭警示 (預測已降權)'} | 指數: {mkt_p:.0f}\n"
    msg += "----------------------------------\n"
    
    for i, s in enumerate(top_keys):
        r = results[s]
        rs_label = "強於大盤" if r['rs'] > 0 else "弱於大盤"
        msg += f"{['🥇','🥈','🥉','📈','📈'][i]} **{s}** 預估 `{r['p']:+.2%}`\n"
        msg += f"   現價: `{r['c']:.1f}` | {rs_label}\n"

    safe_post(msg[:1900])

# =========================
# 輔助函數 (選股池)
# =========================
def get_tw_pool():
    try:
        res = requests.get("https://isin.twse.com.tw/isin/C_public.jsp?strMode=2", timeout=10)
        df = pd.read_html(res.text)[0]
        df.columns = df.iloc[0]; df = df.iloc[1:]
        df["code"] = df["有價證券代號及名稱"].str.split("　").str[0]
        return [f"{s}.TW" for s in df[df["code"].str.len() == 4]["code"].tolist()[:300]]
    except: return ["2330.TW", "2317.TW"]

def get_us_pool():
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=headers, timeout=10)
        df = pd.read_html(res.text)[0]
        return [s.replace('.', '-') for s in df['Symbol'].tolist()[:300]]
    except: return ["AAPL", "NVDA", "TSLA"]

def safe_post(msg):
    if not WEBHOOK_URL: print(msg); return
    try: requests.post(WEBHOOK_URL, json={"content": msg}, timeout=15)
    except: pass

if __name__ == "__main__":
    # 執行台股分析，若要跑美股可改為 run("US")
    run("TW")
