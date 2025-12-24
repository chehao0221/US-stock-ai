🚀 US Stock AI Predictor (美股 AI 自動化分析系統) 專為美股市場設計的自動化分析工具。每日美股收盤後，自動掃描 S&P 500 指數成分股，利用 AI 模型識別具備短期爆發力的標的，並監控科技巨頭 (Magnificent 7) 的走勢變化。

🌟 核心功能

S&P 500 海選：每日台北時間 06:00 (美股收盤後) 自動運行，從 S&P 500 中海選 5 檔預期報酬最優標的。

科技巨頭監控：固定監測 AAPL、NVDA、TSLA、MSFT、GOOGL、AMZN、META 之 AI 預估趨勢。

美股特性優化：針對美股波動率調整特徵權重，並提供小數點後兩位的精確支撐壓力位。

自動化回測：具備 us_history.csv 數據庫，自動進行 5 日循環的勝率結算。

🛠️ 技術棧

語言: Python 3.10

數據來源: yfinance (S&P 500 Components)

自動化: GitHub Actions (台北時間 06:00 AM)

📊 報表示例

🇺🇸 美股 AI 進階預測報告 (2025-12-24) 🏆 AI 海選 Top 5 (美股潛力股) 🥇 NVDA: 預估 +3.45% └ 現價: 145.20 (支撐: 138.50 / 壓力: 152.30) ... 💎 科技巨頭監控 (Magnificent 7) TSLA: 預估 -1.20% | 現價: 350.10



🚀 快速開始 
設定 Discord Webhook：在您的 Discord 頻道建立 Webhook 並複製網址。

GitHub Secrets 設定：

DISCORD_WEBHOOK_URL：放入您的 Webhook 網址。

權值設定：

前往 GitHub 設定 Workflow permissions 為 Read and write permissions，以確保歷史紀錄可存檔。

⚠️ 免責聲明 本專案僅供機器學習研究參考，不構成任何投資建議。股市投資有風險，AI 預測可能存在誤差，請投資人審慎評估。
