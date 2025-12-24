import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
from xgboost import XGBRegressor
from datetime import datetime
import warnings

# =========================
# 基本設定
# =========================
warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(BASE_DIR, "us_history.csv")
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

# =========================
# 股票池 (S&P 500 前 300 檔)
# =========================
def get_us_300_pool():
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=headers, timeout=10)
        df = pd.read_html(res.text)[0]
        # 修正美股代碼中的點（如 BRK.B 改為 BRK-B）以符合 yfinance 格式
        symbols = [s.replace('.', '-') for s in df['Symbol'].tolist()]
        return symbols[:300]
    except Exception as e:
        print(f"池化抓取失敗: {e}")
        return ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "AMZN", "META"]

# =========================
# 大盤環境監測 (S&P 500)
# =========================
def get_market_context():
    try:
        idx = yf.download("^GSPC", period="1y", auto_adjust=True, progress=False)
        if idx.empty: return True, 0, 0, None
        idx["ma60"] = idx["Close"].rolling(60).mean()
        curr_p = float(idx["Close"].iloc[-1])
        ma60_p = float(idx["ma60"].iloc[-1])
        return (curr_p > ma60_p), curr_p, ma60_p, idx
    except:
        return True, 0, 0, None

# =========================
# 進階特徵工程
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
    
    # 3. ATR 波動率指標
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    df["atr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    
    # 4. 相對強度 (RS Index) - 相對於 S&P 500
    if market_df is not None:
        mkt_ret = market_df["Close"].pct_change(20)
        df["rs_index"] = df["Close"].pct_change(20) - mkt_ret.reindex(df.index)
    else:
        df["rs_index"] = 0
    
    # 5. 成交金額 (流動性)
    df["avg_amount"] = (df["Close"] * df["Volume"]).rolling(5).mean()
    return df

# =========================
# 紀錄與對帳
# =========================
def audit_and_save(results, top_keys):
    if os.path.exists(HISTORY_FILE):
        hist = pd.read_csv(HISTORY_FILE)
        hist["date"] = pd.to_datetime(hist["date"]).dt.date
    else:
        hist = pd.DataFrame(columns=["date", "symbol", "pred_p", "pred_ret", "settled"])
    
    today = datetime.now().date()
    new_rows = []
    for s in top_keys:
        if results[s]["c"] <= 0: continue
        new_rows.append({
            "date": today,
            "symbol": s,
            "pred_p": results[s]["c"],      # 紀錄現價供結算
            "pred_ret": results[s]["p"],    # 紀錄 AI 預測漲幅
            "settled": False
        })
    
    if new_rows:
        hist = pd.concat([hist, pd.DataFrame(new_rows)], ignore_index=True)
        hist = hist.drop_duplicates(subset=["date", "symbol"], keep="last")
        hist.to_csv(HISTORY_FILE, index=False)

# =========================
# 主分析流程
# =========================
def run():
    is_bull, mkt_p, mkt_ma, mkt_df = get_market_context()
    must_watch = ["AAPL", "NVDA", "TSLA", "MSFT"]
    pool = get_us_300_pool()
    watch = list(set(must_watch + pool))
    
    print(f"🚀 美股 AI 分析啟動 | 市場趨勢：{'多頭' if is_bull else '空頭（防禦模式）'}")
    
    # 一次性抓取 300 檔標的數據
    all_data = yf.download(watch, period="5y", group_by="ticker", auto_adjust=True, progress=False)
    
    feats = ["mom20", "rsi", "bias", "vol_ratio", "rs_index"]
    results = {}
    MIN_AMOUNT = 10_000_000 # 門檻：1000萬美金

    for s in watch:
        try:
            if s not in all_data or all_data[s].empty: continue
            
            df = all_data[s].dropna()
            if len(df) < 150: continue
            
            df = compute_features(df, market_df=mkt_df)
            last = df.iloc[-1]
            
            # 1. 流動性濾網
            if last["avg_amount"] < MIN_AMOUNT: continue

            # 2. 訓練集準備 (近 500 根 K 線)
            df["target"] = df["Close"].shift(-5) / df["Close"] - 1
            train = df.dropna().iloc[-500:]
            
            if len(train) < 100: continue

            # 3. 模型訓練
            model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
            model.fit(train[feats], train["target"])
            
            # 4. 預測
            pred = float(np.clip(model.predict(train[feats].iloc[-1:])[0], -0.15, 0.15))
            
            # 5. 風險干預
            if not is_bull: pred *= 0.5  # 大盤空頭降權
            if last["atr"] > df["atr"].mean() * 1.5: pred *= 0.8  # 高波動降權
            if pred < 0.01: pred = 0  # 噪音保護

            results[s] = {
                "p": pred,
                "c": float(last["Close"]),
                "rs": float(last["rs_index"])
            }
        except:
            continue

    # 排序選出 Top 5 (排除固定監測股)
    horses = {k: v for k, v in results.items() if k not in must_watch}
    top_keys = sorted(horses, key=lambda x: horses[x]["p"], reverse=True)[:5]
    final_keys = [k for k in top_keys if horses[k]["p"] > 0]

    # 儲存與對帳
    audit_and_save(results, final_keys)

    # 報告組裝
    msg = f"🇺🇸 **美股 AI 進階預報 ({datetime.now():%m/%d})**\n"
    msg += f"{'📈 多頭環境' if is_bull else '⚠️ 空頭警示 (預測已降權)'} | 指數: {mkt_p:.1f}\n"
    msg += "----------------------------------\n"
    
    if not final_keys:
        msg += "💡 市場信號不足，建議保守觀望。\n"
    else:
        for i, s in enumerate(final_keys):
            r = results[s]
            msg += f"{['🥇','🥈','🥉','📈','📈'][i]} **{s}** 預估 `{r['p']:+.2%}` | RS:{'強' if r['rs']>0 else '弱'}\n"

    # 權值監測
    msg += "\n🔍 **權值/監測標的**\n"
    for s in must_watch:
        if s in results:
            msg += f"`{s}` 預估 `{results[s]['p']:+.2%}`\n"
    
    if WEBHOOK_URL:
        try:
            requests.post(WEBHOOK_URL, json={"content": msg[:1900]}, timeout=15)
        except:
            print("Webhook 傳送失敗")
    else:
        print(msg)

if __name__ == "__main__":
    run()
