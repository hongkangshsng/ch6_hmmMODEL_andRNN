# 📊 機器學習訓練資料預處理管道 (Data Preprocessing Pipeline)

本專案實作了一個完整的 Python 自動化資料預處理流程。透過 `scikit-learn` 的 `Pipeline` 與 `ColumnTransformer` 技術，實現包含缺失值處理（Imputation）、特徵縮放（Feature Scaling）、獨熱編碼（One-Hot Encoding）以及資料分佈視覺化的完整運作機制，適用於機器學習模型訓練前的資料準備階段。

---

## 🛠️ 預處理執行步驟 (Preprocessing Pipeline)

在將資料送入機器學習模型前，我們執行了以下核心預處理步驟：

1. **缺失值處理 (Missing Value Imputation)**：
   * **數值型特徵** (`Age`, `Salary`, `Experience_Years`)：採用 **中位數 (Median)** 填補，避免極端數值（離群值）拉偏整體數據。
   * **類別型特徵** (`Department`, `City`, `Education`)：採用 **眾數 (Most Frequent)** 填補最常見的類別。

2. **特徵縮放 (Feature Scaling)**：
   * 使用 **StandardScaler (Z-score 標準化)**，將數值特徵轉換為平均值為 0、標準差為 1 的常態分佈，消除因不同變數單位（如薪資與年齡）數量級差異過大對模型權重產生的偏誤。

3. **類別轉換 (Categorical Encoding)**：
   * 使用 **One-Hot Encoding (獨熱編碼)** 將類別文字轉為二元數值矩陣（0 或 1），避免給予類別字串不合理的順序大小關聯。

4. **資料集劃分與防止洩漏 (Train/Test Split & Data Leakage Prevention)**：
   * 先將資料以 8:2 比例劃分為訓練集與測試集。
   * **僅對訓練集執行 `.fit_transform()`**，測試集則使用訓練集擬合出的參數執行 `.transform()`，確保訓練過程完全符合真實環境嚴謹度。

---

## 📈 視覺化分析報告 (Visualization Analysis)

下圖為預處理管道執行後導出的數據特徵與轉換結構圖：

![Data Preprocessing Results](preprocessing_result.png)

### 1. 數值特徵縮放分佈圖 (圖左)
* **圖表意涵**：展示原始薪資與標準化後（Z-score）薪資的分佈重疊比較。
* **數據解析**：
  * **下軸 (Salary)**：原始薪資數據涵蓋範圍廣（$20,000 \sim 120,000$）。
  * **上軸 (Scaled Salary)**：經過 `StandardScaler` 轉換後，數據中心點縮放至 **0** 附近，絕大多數數據落在 **-2 至 +2** 的標準差區間內。
  * **核心效果**：成功保留了原始薪資右偏分佈（Right-skewed）的數據型態，同時縮放數值範圍以提升模型的梯度下降收斂效率。

### 2. 類別特徵獨熱編碼矩陣 (圖中)
* **圖表意涵**：抽樣展示前 15 筆樣本經 One-Hot Encoding 後的二元值（0 與 1）分佈熱力圖。
* **數據解析**：
  * **X 軸**：類別文字欄位被拆解為獨立特徵（例如 `Department_HR`、`City_Taipei`、`Education_Master` 等）。
  * **Y 軸 (Sample Index)**：代表具體的樣本編號。
  * **黃色 (0) / 深藍色 (1)**：清楚呈現每個樣本在對應類別特徵上的激活狀態，確保類別特徵轉化為模型可直接計算的矩陣。

### 3. 特徵相關性矩陣 (圖右)
* **圖表意涵**：呈現預處理後所有特徵變數之間的皮爾森相關係數 (Pearson Correlation Matrix)。
* **數據解析**：
  * 對角線呈現深紅（相關性為 1.0），代表變數與自身高度相關。
  * **多重共線性檢視**：圖中未出現大面積深紅（高正相關）或深藍（高負相關）的異常區塊，說明經 One-Hot Encoding 與縮放後的特徵之間獨立性良好，無嚴重的共線性（Multicollinearity）問題，非常適合直接輸入線性模型或樹狀模型進行訓練。

---

## 📁 檔案結構 (Project Structure)

```text
├── training_data.csv          # 原始訓練數據資料集
├── 訓練樣本預處理.py           # 預處理與視覺化主要執行程式碼
├── preprocessing_result.png   # 導出的預處理視覺化分析圖表
└── README.md                  # 專案說明文件