import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hmmlearn.hmm import CategoricalHMM

# 設定繪圖風格與字型
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

# ==========================================
# 1. 讀取與處理 200 筆訓練資料集
# ==========================================
file_path = "training_data.csv"
df = pd.read_csv(file_path)

states = ["Pending", "Under_Review", "Approved", "Rejected"]
n_states = len(states)

# 依薪資與年齡簡化風險等級觀測值 (0: 低風險, 1: 中風險, 2: 高風險)
df["Risk_Level"] = 1
df.loc[(df["Salary"] > 70000) & (df["Age"] > 30), "Risk_Level"] = 0
df.loc[(df["Salary"] < 45000) | (df["Age"] < 25), "Risk_Level"] = 2

np.random.seed(42)
simulated_state_sequences = []
simulated_obs_sequences = []

for _, row in df.iterrows():
    risk = row["Risk_Level"]
    if row["Loan_Approved"] == 1:
        s_seq = [0, 1, 2]  # Pending -> Under_Review -> Approved
    else:
        s_seq = [0, 1, 3]  # Pending -> Under_Review -> Rejected

    simulated_state_sequences.append(s_seq)
    simulated_obs_sequences.append([risk, risk, risk])

# ==========================================
# 2. 統計 HMM 三要素矩陣 (pi, A, B)
# ==========================================
# (1) 初始狀態概率向量 pi
pi = np.zeros(n_states)
for seq in simulated_state_sequences:
    pi[seq[0]] += 1
pi = pi / np.sum(pi)

# (2) 狀態轉移矩陣 A (修復終點狀態列和為 1)
A = np.zeros((n_states, n_states))
for seq in simulated_state_sequences:
    for t in range(len(seq) - 1):
        A[seq[t], seq[t + 1]] += 1

row_sums = A.sum(axis=1, keepdims=True)
for i in range(n_states):
    if row_sums[i] == 0:
        A[i, i] = 1.0
        row_sums[i] = 1.0
A = A / row_sums

# (3) 觀測(發射)概率矩陣 B
n_obs = 3
B = np.zeros((n_states, n_obs))
for s_seq, o_seq in zip(
    simulated_state_sequences, simulated_obs_sequences
):
    for s, o in zip(s_seq, o_seq):
        B[s, o] += 1
b_row_sums = B.sum(axis=1, keepdims=True)
b_row_sums[b_row_sums == 0] = 1.0
B = B / b_row_sums

# ==========================================
# 3. 建立並執行 CategoricalHMM 解碼模型
# ==========================================
model = CategoricalHMM(n_components=n_states)
model.startprob_ = pi
model.transmat_ = A
model.emissionprob_ = B

# 測試序列輸入
test_obs = np.array([[0], [1], [2]])  # 低風險 -> 中風險 -> 高風險
_, estimated_states = model.decode(test_obs, algorithm="viterbi")
predicted_state_names = [states[s] for s in estimated_states]

# ==========================================
# 4. HMM 統計數據視覺化 (導出 hmm_result.png)
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 圖 1：狀態轉移矩陣 A 熱力圖
sns.heatmap(
    A,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    xticklabels=states,
    yticklabels=states,
    ax=axes[0],
    cbar=False,
)
axes[0].set_title("1. 狀態轉移概率矩陣 (Transition Matrix A)", fontsize=12)
axes[0].set_xlabel("To State (下一狀態)")
axes[0].set_ylabel("From State (當前狀態)")

# 圖 2：發射(觀測)概率矩陣 B 熱力圖
obs_labels = ["Low_Risk(0)", "Mid_Risk(1)", "High_Risk(2)"]
sns.heatmap(
    B,
    annot=True,
    fmt=".2f",
    cmap="Oranges",
    xticklabels=obs_labels,
    yticklabels=states,
    ax=axes[1],
    cbar=False,
)
axes[1].set_title(
    "2. 觀測(發射)概率矩陣 (Emission Matrix B)", fontsize=12
)
axes[1].set_xlabel("Observation (風險特徵)")
axes[1].set_ylabel("State (審核階段)")

plt.tight_layout()
plt.savefig("hmm_result.png", dpi=300, bbox_inches="tight")
print("6-2 HMM 執行成功！視覺化圖表已成功儲存為 hmm_result.png！")
print(
    f"維特比解碼測試結果：[0, 1, 2] -> {' -> '.join(predicted_state_names)}"
)