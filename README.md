# 📊 機器學習訓練資料預處理管道 (Data Preprocessing Pipeline)

# 📊 機器學習數據處理與隱馬爾可夫模型實務 (ML Preprocessing & HMM)

本專案包含兩個核心單元：**自動化資料預處理管道**與**隱馬爾可夫模型 (HMM) 狀態解碼**。整體流程基於 200 筆貸款申請數據，展示了從原始資料清洗、特徵工程到時序狀態序列預測的完整機器學習開發實務[cite: 8, 21]。

---

## 🛠️ 第一節：機器學習訓練資料預處理管道 (6-1)

在將資料送入機器學習模型前，我們透過 `scikit-learn` 的 `Pipeline` 與 `ColumnTransformer` 技術執行以下核心步驟[cite: 8]：

1. **缺失值處理 (Missing Value Imputation)**：
   * **數值型欄位** (`Age`, `Salary`, `Experience_Years`)：採用 **中位數 (Median)** 填補，避免離群值拉偏整體數據[cite: 8]。
   * **類別型欄位** (`Department`, `City`, `Education`)：採用 **眾數 (Most Frequent)** 填補最常見的類別[cite: 8]。
2. **特徵縮放 (Feature Scaling)**：
   * 使用 **StandardScaler (Z-score 標準化)**，將數值特徵轉換為平均值為 0、標準差為 1 的常態分佈[cite: 8]。
3. **類別轉換 (Categorical Encoding)**：
   * 使用 **One-Hot Encoding (獨熱編碼)** 將類別文字轉為二元數值矩陣（0 或 1）[cite: 8]。
4. **防止資料洩漏 (Data Leakage Prevention)**：
   * 劃分 8:2 訓練/測試集，且**僅對訓練集執行 `.fit_transform()`**[cite: 8]。

### 📈 預處理視覺化分析報告

![Data Preprocessing Results](preprocessing_result.png)

* **圖 1 (數值特徵縮放分佈)**：原始薪資數據（$20,000 \sim 120,000$）被成功收斂至 Z-score 範圍（-2 至 +2）內，加速模型梯度下降效率[cite: 6]。
* **圖 2 (獨熱編碼矩陣)**：展示類別變數轉化為二元 0/1 矩陣的激活狀態[cite: 6]。
* **圖 3 (特徵相關性矩陣)**：各欄位相關性良好，無多重共線性（Multicollinearity）問題[cite: 6]。

---

## 🔄 第二節：隱馬爾可夫模型 (HMM) 統計與狀態解碼 (6-2)

在第二階段，我們採用 **隱馬爾可夫模型 (Hidden Markov Model, HMM)**，將客戶從申請到審核完成的動態過程進行時序建模[cite: 17, 21]。

### 📘 HMM 三要素統計概念
* **初始狀態概率向量 ($\pi$)**：客戶進入審核流程的起始狀態分佈（100% 從 `Pending` 開始）[cite: 17, 21]。
* **狀態轉移概率矩陣 ($A$)**：計算客戶從當前審核狀態轉移到下一個狀態的概率（如 `Pending` $\rightarrow$ `Under_Review`）[cite: 17, 21]。
* **觀測概率矩陣 ($B$) / 發射矩陣**：計算特定審核狀態下，表現出不同特徵風險等級（高/中/低風險）的條件概率[cite: 17, 21]。

### 📊 HMM 統計矩陣視覺化分析

![HMM Matrix Results](hmm_result.png)

#### 1. 狀態轉移概率矩陣 (圖左)
* **圖表意涵**：呈現隱藏狀態（Pending, Under_Review, Approved, Rejected）之間的轉移機率[cite: 21]。
* **數據解析**：
  * `Pending` 到 `Under_Review` 的轉移概率為 **1.00**，說明所有申請案皆必經審核階段[cite: 21]。
  * `Under_Review` 依據客戶風險特徵，分流轉移至 `Approved` (核准, 0.35) 或 `Rejected` (退件, 0.65)[cite: 21]。
  * `Approved` 與 `Rejected` 作為審核終點狀態（吸收態），其轉移矩陣列和經修復後嚴格符合機率公理（等於 1.00）[cite: 19, 21]。

#### 2. 觀測概率矩陣 (圖右)
* **圖表意涵**：呈現特定隱藏狀態發射出對應風險特徵（0: 低風險, 1: 中風險, 2: 高風險）的條件機率分佈[cite: 21]。
* **數據解析**：
  * 透過 **`CategoricalHMM`** 與 **維特比演算法 (Viterbi Algorithm)**，模型可成功從連續觀測風險序列中，解碼出推測概率最高的隱藏審核狀態變化鏈[cite: 18, 20]。

---

## 📁 專案檔案結構 (Project Structure)

```text
├── training_data.csv          # 200 筆原始訓練數據集
├── 訓練樣本預處理.py           # 6-1 節預處理與視覺化程式碼
├── 6-2_hmm_model.py           # 6-2 節 HMM 統計矩陣與維特比解碼程式碼
├── preprocessing_result.png   # 6-1 節預處理視覺化圖表
├── hmm_result.png             # 6-2 節 HMM 矩陣視覺化圖表
└── README.md                  # 專案完整說明文件