from utils.market_calendar import is_market_open
from utils.safe_yfinance import safe_yf_download

def pre_check():
    if not is_market_open("US"):
        print("📌 美股未開盤")
        return False
    return True

import pandas as pd
import requests
import os
from xgboost import XGBRegressor
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(BASE_DIR, "us_history.csv")
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

MAG7 = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "AMZN", "META"]

def calc_pivot(df):
    r = df.iloc[-20:]
    h, l, c = r["High"].max(), r["Low"].min(), r["Close"].iloc[-1]
    p = (h + l + c) / 3
    return round(2*p - h, 2), round(2*p - l, 2)

def get_top300_by_volume():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    df = pd.read_html(requests.get(url, headers=headers, timeout=10).text)[0]
    tickers = [s.replace(".", "-") for s in df["Symbol"]]

    vol_data = safe_yf_download(tickers, period="1mo", max_chunk=80)
    avg_vol = {
        t: v["Volume"].tail(20).mean()
        for t, v in vol_data.items()
        if "Volume" in v
    }

    return sorted(avg_vol, key=avg_vol.get, reverse=True)[:300]

def run():
    universe = list(dict.fromkeys(MAG7 + get_top300_by_volume()))
    data = safe_yf_download(universe, period="2y", max_chunk=80)

    feats = ["mom20", "bias", "vol_ratio"]
    results = {}

    for s, df in data.items():
        if len(df) < 160:
            continue

        df["mom20"] = df["Close"].pct_change(20)
        df["bias"] = (df["Close"] - df["Close"].rolling(20).mean()) / df["Close"].rolling(20).mean()
        df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
        df["target"] = df["Close"].shift(-5) / df["Close"] - 1

        train = df.iloc[:-5].dropna()
        if len(train) < 80:
            continue

        model = XGBRegressor(
            n_estimators=90,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(train[feats], train["target"])

        pred = float(model.predict(df[feats].iloc[-1:])[0])
        sup, res = calc_pivot(df)

        results[s] = {
            "pred": pred,
            "price": round(df["Close"].iloc[-1], 2),
            "sup": sup,
            "res": res
        }

    msg = f"📊 **美股 AI 進階預測報告 ({datetime.now():%Y-%m-%d})**\n"
    msg += "------------------------------------------\n\n"

    medals = ["🥇", "🥈", "🥉", "📈", "📈"]
    horses = {k: v for k, v in results.items() if k not in MAG7 and v["pred"] > 0}
    top5 = sorted(horses, key=lambda x: horses[x]["pred"], reverse=True)[:5]

    msg += "🏆 **AI 海選 Top 5 (潛力股)**\n"
    for i, s in enumerate(top5):
        r = results[s]
        msg += f"{medals[i]} {s}: 預估 `{r['pred']:+.2%}`\n"
        msg += f" └ 現價: `{r['price']}` (支撐: `{r['sup']}` / 壓力: `{r['res']}`)\n"

    msg += "\n💎 **Magnificent 7 監控 (固定顯示)**\n"
    for s in MAG7:
        if s in results:
            r = results[s]
            msg += f"{s}: 預估 `{r['pred']:+.2%}`\n"
            msg += f" └ 現價: `{r['price']}` (支撐: `{r['sup']}` / 壓力: `{r['res']}`)\n"

    msg += "\n🏁 美股 5 日回測結算報告\n\n💡 AI 為機率模型，僅供研究參考"

    if WEBHOOK_URL:
        requests.post(WEBHOOK_URL, json={"content": msg[:1900]})
    else:
        print(msg)

if __name__ == "__main__":
    if pre_check():
        run()
