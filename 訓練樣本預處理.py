"""Preprocess the bundled loan data and save reproducible visualizations."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Allow image generation in headless environments.

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "training_data.csv"
OUTPUT_PATH = BASE_DIR / "preprocessing_result.png"
TARGET_COLUMN = "Loan_Approved"
NUMERIC_FEATURES = ["Age", "Salary", "Experience_Years"]
CATEGORICAL_FEATURES = ["Department", "City", "Education"]
REQUIRED_COLUMNS = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN])


def load_data(data_path: Path) -> pd.DataFrame:
    """Load the bundled CSV and fail early when its schema is incompatible."""
    df = pd.read_csv(data_path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"training_data.csv 缺少必要欄位：{sorted(missing)}")
    return df


def make_preprocessor() -> ColumnTransformer:
    """Build the numeric and categorical transformations used in the project."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )


def plot_preprocessing_result(
    raw_training_data: pd.DataFrame,
    processed_training_data: pd.DataFrame,
    encoded_category_columns: list[str],
) -> None:
    """Create the three visualizations used in the README."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.kdeplot(
        raw_training_data["Salary"].dropna(),
        ax=axes[0],
        color="blue",
        label="Raw salary",
        fill=True,
    )
    axes[0].set_title("1. Raw salary distribution", fontsize=12)
    axes[0].set_xlabel("Salary")
    axes[0].legend()

    scaled_axis = axes[0].twiny()
    sns.kdeplot(
        processed_training_data["Salary"],
        ax=scaled_axis,
        color="red",
        label="Scaled salary",
        fill=True,
        alpha=0.3,
    )
    scaled_axis.set_xlabel("Scaled salary (Z-score)")
    scaled_axis.legend(loc="upper right")

    sns.heatmap(
        processed_training_data[encoded_category_columns].head(15),
        annot=True,
        cmap="YlGnBu",
        cbar=False,
        ax=axes[1],
        fmt=".0f",
    )
    axes[1].set_title("2. One-hot encoded categorical features", fontsize=12)
    axes[1].set_ylabel("Sample index")

    correlation = processed_training_data.corr()
    sns.heatmap(correlation, cmap="coolwarm", ax=axes[2], vmin=-1, vmax=1)
    axes[2].set_title("3. Processed feature correlation", fontsize=12)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_data(DATA_PATH)
    features = df.drop(columns=[TARGET_COLUMN])
    target = df[TARGET_COLUMN]

    x_train, x_test, _, _ = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )
    preprocessor = make_preprocessor()

    # Fit only on the training split, then transform train and test data.
    x_train_processed = preprocessor.fit_transform(x_train)
    x_test_processed = preprocessor.transform(x_test)

    category_encoder = preprocessor.named_transformers_["cat"]["onehot"]
    encoded_category_columns = list(
        category_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    )
    feature_names = NUMERIC_FEATURES + encoded_category_columns
    x_train_df = pd.DataFrame(
        x_train_processed, columns=feature_names, index=x_train.index
    )

    plot_preprocessing_result(x_train, x_train_df, encoded_category_columns)
    print(
        "Preprocessing completed: "
        f"{features.shape[1]} input features -> {x_train_df.shape[1]} transformed features."
    )
    print(f"Train/test rows: {len(x_train)}/{len(x_test)}")
    print(f"Transformed test shape: {x_test_processed.shape}")
    print(f"Saved visualization: {OUTPUT_PATH.name}")


if __name__ == "__main__":
    main()
