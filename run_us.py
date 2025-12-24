import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
from xgboost import XGBRegressor
from datetime import datetime, timedelta
import warnings

# =========================
# 基本設定 (美股)
# =========================
warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE_US = os.path.join(BASE_DIR, "us_history.csv")
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

# =========================
# 工具函數：支撐壓力與股票池
# =========================
def calc_sup_res_us(df):
    try:
        recent = df.iloc[-20:]
        h, l, c = recent['High'].max(), recent['Low'].min(), recent['Close'].iloc[-1]
        p = (h + l + c) / 3
        return round(2*p - h, 2), round(2*p - l, 2) # 美股顯示到小數兩位
    except: return 0, 0

def get_us_pool():
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        res = requests.get(url, headers=headers, timeout=10)
        df = pd.read_html(res.text)[0]
        # yfinance 標點符號處理 (如 BRK.B -> BRK-B)
        return [s.replace('.', '-') for s in df['Symbol'].tolist()[:300]]
    except: return ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "AMZN", "META"]

# =========================
# 美股 5 日回測結算
# =========================
def get_us_settle_report():
    if not os.path.exists(HISTORY_FILE_US): return ""
    try:
        df = pd.read_csv(HISTORY_FILE_US)
        df['date'] = pd.to_datetime(df['date'])
        mask = (df['settled'].astype(str).str.upper() == 'FALSE') & (df['date'] <= datetime.now() - timedelta(days=5))
        to_settle = df[mask].copy()
        if to_settle.empty: return "\n📊 **5日回測**: 尚無待結算數據。"

        report = "\n🏁 **美股 5 日回測結算報告**\n"
        syms = to_settle['symbol'].unique().tolist()
        prices = yf.download(syms, period="5d", auto_adjust=True, progress=False)['Close']
        
        for idx, row in to_settle.iterrows():
            s = row['symbol']
            try:
                curr_p = float(prices[s].dropna().iloc[-1]) if isinstance(prices, pd.DataFrame) else float(prices.iloc[-1])
                ret = (curr_p - row['pred_p']) / row['pred_p']
                win = (ret > 0 and row['pred_ret'] > 0) or (ret < 0 and row['pred_ret'] < 0)
                df.at[idx, 'settled'] = 'True'
                report += f"• `{s}`: 預估 {row['pred_ret']:+.2%} | 實際 `{ret:+.2%}` {'✅' if win else '❌'}\n"
            except: continue
        df.to_csv(HISTORY_FILE_US, index=False)
        return report
    except: return ""

# =========================
# 主程序
# =========================
def run_us():
    mag_7 = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "AMZN", "META"]
    pool = get_us_pool()
    watch = list(dict.fromkeys(mag_7 + pool))
    
    print(f"🇺🇸 開始海選 {len(watch)} 檔美股標的...")
    data = yf.download(watch, period="2y", auto_adjust=True, group_by="ticker", progress=False)
    
    results = {}
    feats = ["mom20", "bias", "vol_ratio"]
    
    for s in watch:
        try:
            df = data[s].dropna()
            if len(df) < 50: continue
            df["mom20"] = df["Close"].pct_change(20)
            df["bias"] = (df["Close"] - df["Close"].rolling(20).mean()) / df["Close"].rolling(20).mean()
            df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
            df["target"] = df["Close"].shift(-5) / df["Close"] - 1
            
            train = df.dropna().iloc[-300:]
            model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05).fit(train[feats], train["target"])
            pred = float(model.predict(df[feats].iloc[-1:])[0])
            sup, res = calc_sup_res_us(df)
            results[s] = {"p": pred, "c": float(df["Close"].iloc[-1]), "sup": sup, "res": res}
        except: continue

    # 組合訊息 (排版比照圖二)
    msg = f"🇺🇸 **美股 AI 進階預測報告 ({datetime.now():%Y-%m-%d})**\n"
    msg += "------------------------------------------\n\n"
    msg += "🏆 **AI 海選 Top 5 (美股潛力股)**\n"
    
    horses = {k: v for k, v in results.items() if k not in mag_7}
    top_5 = sorted(horses, key=lambda x: horses[x]["p"], reverse=True)[:5]
    
    for i, s in enumerate(top_5):
        r = results[s]
        msg += f"{['🥇','🥈','🥉','📈','📈'][i]} **{s}**: 預估 `{r['p']:+.2%}`\n"
        msg += f" └ 現價: `{r['c']:.2f}` (支撐: `{r['sup']}` / 壓力: `{r['res']}`)\n"

    msg += "\n💎 **科技巨頭監控 (Magnificent 7)**\n"
    for s in mag_7:
        if s in results:
            r = results[s]
            msg += f"**{s}**: 預估 `{r['p']:+.2%}`\n └ 現價: `{r['c']:.2f}`\n"

    # 加上回測報告
    msg += get_us_settle_report()
    msg += "\n💡 AI 預測僅供參考，美股波動大請注意風險。"

    if WEBHOOK_URL:
        requests.post(WEBHOOK_URL, json={"content": msg[:1900]}, timeout=15)
    else: print(msg)

    # 存檔供未來結算
    new_hist = [{"date": datetime.now().date(), "symbol": s, "pred_p": results[s]['c'], "pred_ret": results[s]['p'], "settled": "False"} for s in (top_5 + mag_7) if s in results]
    pd.DataFrame(new_hist).to_csv(HISTORY_FILE_US, mode='a', header=not os.path.exists(HISTORY_FILE_US), index=False)

if __name__ == "__main__":
    run_us()
