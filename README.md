 US Stock AI Predictor (美股 AI 自動化預測)
這是一個針對美股市場開發的自動化預測系統。它結合了 XGBoost 機器學習模型 與 GitHub Actions，每天自動抓取美股數據、進行模型訓練，並透過 Discord Webhook 推播高潛力標的與預測分析報告。

🌟 核心功能

全自動化排程：每日美股收盤後，自動於 GitHub Actions 虛擬環境執行 。

多維度特徵工程：整合 RSI 強弱指標、20 日均線乖離率、成交量變化及價格動量指標。

支撐與壓力計算：基於 60 日歷史高低點，自動標註技術面支撐與壓力位。

精準對帳回測：系統會自動追蹤 5 天前的預測紀錄，並根據實際收盤價計算勝率與回報誤差。

Discord 即時推播：將預測結果、現價、技術支撐壓力及歷史對帳清單整理成精美報表。

🛠️ 技術架構

執行環境: Python 3.10 (GitHub Actions) 

核心模型: XGBoost Regressor

數據來源: yfinance (Yahoo Finance API)

依賴套件: pandas, numpy, scikit-learn, requests 等

追蹤指標:

mom20: 20 日動量

rsi: 相對強弱指標

bias: 均線乖離率

vol_ratio: 成交量比率

🚀 部署指南
1. 設置環境變數 (Secrets)
為了讓腳本能傳送 Discord 通知，請在 GitHub 儲存庫設定：

前往 Settings > Secrets and variables > Actions。

新增 Secret：DISCORD_WEBHOOK_URL 並填入你的 Discord Webhook 網址 。

2. 調整觀察清單
你可以直接在 run_us.py 中的 watch 列表修改你想追蹤的標的（如：AAPL, TSLA, NVDA, MSFT 等）。

3. 自動執行時間
目前設定為 UTC 時間 02:00 運行（約為美股收盤後），對應台北時間為上午 10:00 。

📊 報告內容示例
📊 美股 AI 進階預測報告 (2025-XX-XX)
⭐ NVDA 預估 5 日：+3.15% └ 現價 145.20｜支撐 138.50｜壓力 152.00

🎯 5 日預測結算對帳 AAPL +1.20% ➜ +1.45% ✅ TSLA +2.50% ➜ -0.80% ❌

📂 檔案結構描述
run_us.py: 主程式，包含數據抓取、模型訓練與 Discord 發送邏輯。

us_history.csv: 自動保存預測紀錄，用於後續對帳與成效追蹤。


.github/workflows/us_analysis.yml: 定義自動化排程與環境部署步驟 。

requirements.txt: 專案所需的 Python 函式庫清單。

⚠️ 投資風險聲明
本專案係利用歷史數據進行統計模擬，不構成任何形式的投資建議。美股波動劇烈，模型預測僅供參考，實際投資請務必獨立判斷並承擔風險。
