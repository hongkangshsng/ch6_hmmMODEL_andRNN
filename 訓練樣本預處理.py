import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 設定繪圖風格與中文字型（若無中文字型可自動降級顯示）
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

# ==========================================
# 1. 讀取數據與基礎預處理
# ==========================================
file_path = "training_data.csv"
df = pd.read_csv(file_path)

X = df.drop(columns=["Loan_Approved"])
y = df["Loan_Approved"]

numeric_features = ["Age", "Salary", "Experience_Years"]
categorical_features = ["Department", "City", "Education"]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_processed = preprocessor.fit_transform(X_train)

# 取得轉換後的完整特徵欄位名稱
cat_encoder = preprocessor.named_transformers_["cat"]["onehot"]
encoded_cat_cols = cat_encoder.get_feature_names_out(categorical_features)
all_feature_names = numeric_features + list(encoded_cat_cols)

X_train_df = pd.DataFrame(
    X_train_processed, columns=all_feature_names, index=X_train.index
)

print(
    f"預處理完成！特徵維度從 {X_train.shape[1]} 維擴展為 {X_train_df.shape[1]} 維。"
)

# ==========================================
# 2. 預處理結果視覺化 (Data Visualization)
# ==========================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 圖 1：預處理前後的數值分佈 (以 Salary 為例)
sns.kdeplot(
    X_train["Salary"].dropna(),
    ax=axes[0],
    color="blue",
    label="Raw Salary (原始)",
    fill=True,
)
axes[0].set_title("1. 原始薪資分佈 (含有缺失值與大範圍數值)", fontsize=12)
axes[0].set_xlabel("Salary")
axes[0].legend()

ax0_twin = axes[0].twiny()
sns.kdeplot(
    X_train_df["Salary"],
    ax=ax0_twin,
    color="red",
    label="Scaled Salary (標準化後)",
    fill=True,
    alpha=0.3,
)
ax0_twin.set_xlabel("Scaled Salary (Z-score)")
ax0_twin.legend(loc="upper right")

# 圖 2：類別特徵經 One-Hot Encoding 後的熱力圖 (前 15 筆)
sns.heatmap(
    X_train_df[encoded_cat_cols].head(15),
    annot=True,
    cmap="YlGnBu",
    cbar=False,
    ax=axes[1],
    fmt=".0f",
)
axes[1].set_title(
    "2. 類別特徵轉獨熱編碼 (One-Hot Encoded Features)", fontsize=12
)
axes[1].set_ylabel("Sample Index")

# 圖 3：處理後的特徵相關性矩陣 (Correlation Heatmap)
corr = X_train_df.corr()
sns.heatmap(corr, cmap="coolwarm", ax=axes[2], vmin=-1, vmax=1)
axes[2].set_title("3. 預處理後全特徵相關性圖", fontsize=12)

plt.tight_layout()
# 存成高解析度圖片檔案
plt.savefig("preprocessing_result.png", dpi=300, bbox_inches="tight")
print("視覺化圖表已成功儲存為 preprocessing_result.png！")