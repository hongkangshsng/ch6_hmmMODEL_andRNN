"""Estimate a small categorical HMM from the bundled loan-workflow example.

This is a teaching demonstration. The hidden workflow states are constructed
from Loan_Approved and the three-valued observation is a simple rule based
on age and salary; this script is not a trained loan-approval model.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Allow image generation in headless environments.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hmmlearn.hmm import CategoricalHMM


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "training_data.csv"
OUTPUT_PATH = BASE_DIR / "hmm_result.png"
STATES = ["Pending", "Under_Review", "Approved", "Rejected"]
OBSERVATION_LABELS = ["Low_Risk (0)", "Mid_Risk (1)", "High_Risk (2)"]
REQUIRED_COLUMNS = {"Age", "Salary", "Loan_Approved"}


def load_data(data_path: Path) -> pd.DataFrame:
    """Load and validate the tabular data needed by this demonstration."""
    df = pd.read_csv(data_path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"training_data.csv 缺少必要欄位：{sorted(missing)}")
    if not df["Loan_Approved"].isin([0, 1]).all():
        raise ValueError("Loan_Approved 必須只包含 0（Rejected）或 1（Approved）。")
    return df


def add_risk_level(df: pd.DataFrame) -> pd.DataFrame:
    """Add the fixed three-class observation used by the categorical HMM.

    0 = salary > 70,000 and age > 30; 2 = salary < 45,000 or age < 25;
    all remaining cases (including missing values that match neither rule) are 1.
    """
    result = df.copy()
    result["Risk_Level"] = 1
    result.loc[
        (result["Salary"] > 70000) & (result["Age"] > 30), "Risk_Level"
    ] = 0
    result.loc[
        (result["Salary"] < 45000) | (result["Age"] < 25), "Risk_Level"
    ] = 2
    return result


def build_sequences(df: pd.DataFrame) -> tuple[list[list[int]], list[list[int]]]:
    """Construct labelled three-step workflow sequences for each application."""
    state_sequences: list[list[int]] = []
    observation_sequences: list[list[int]] = []

    for _, row in df.iterrows():
        terminal_state = 2 if row["Loan_Approved"] == 1 else 3
        state_sequences.append([0, 1, terminal_state])
        risk = int(row["Risk_Level"])
        # The same applicant-level risk observation is repeated at each step.
        observation_sequences.append([risk, risk, risk])

    return state_sequences, observation_sequences


def estimate_parameters(
    state_sequences: list[list[int]], observation_sequences: list[list[int]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate pi, A and B by normalized counts from labelled sequences."""
    n_states = len(STATES)
    n_observations = len(OBSERVATION_LABELS)
    initial = np.zeros(n_states)
    transition = np.zeros((n_states, n_states))
    emission = np.zeros((n_states, n_observations))

    for state_sequence, observation_sequence in zip(
        state_sequences, observation_sequences
    ):
        initial[state_sequence[0]] += 1
        for current_state, next_state in zip(state_sequence, state_sequence[1:]):
            transition[current_state, next_state] += 1
        for state, observation in zip(state_sequence, observation_sequence):
            emission[state, observation] += 1

    initial /= initial.sum()
    for state_index in range(n_states):
        transition_total = transition[state_index].sum()
        if transition_total == 0:
            # A terminal state has no observed outgoing transition; model it as
            # absorbing so every row remains a valid probability distribution.
            transition[state_index, state_index] = 1.0
        else:
            transition[state_index] /= transition_total

        emission_total = emission[state_index].sum()
        if emission_total == 0:
            # Defensive fallback for an input with no examples of a state.
            emission[state_index] = 1.0 / n_observations
        else:
            emission[state_index] /= emission_total

    return initial, transition, emission


def plot_parameters(transition: np.ndarray, emission: np.ndarray) -> None:
    """Save transition and emission heatmaps alongside this script."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(
        transition,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=STATES,
        yticklabels=STATES,
        ax=axes[0],
        cbar=False,
    )
    axes[0].set_title("State transition matrix A", fontsize=12)
    axes[0].set_xlabel("To state")
    axes[0].set_ylabel("From state")

    sns.heatmap(
        emission,
        annot=True,
        fmt=".2f",
        cmap="Oranges",
        xticklabels=OBSERVATION_LABELS,
        yticklabels=STATES,
        ax=axes[1],
        cbar=False,
    )
    axes[1].set_title("Observation / emission matrix B", fontsize=12)
    axes[1].set_xlabel("Observation")
    axes[1].set_ylabel("State")

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = add_risk_level(load_data(DATA_PATH))
    state_sequences, observation_sequences = build_sequences(df)
    initial, transition, emission = estimate_parameters(
        state_sequences, observation_sequences
    )

    model = CategoricalHMM(
        n_components=len(STATES),
        n_features=len(OBSERVATION_LABELS),
        init_params="",
        params="",
    )
    model.startprob_ = initial
    model.transmat_ = transition
    model.emissionprob_ = emission

    # A fixed toy observation sequence for demonstrating Viterbi decoding.
    test_observations = np.array([[0], [1], [2]])
    _, estimated_states = model.decode(test_observations, algorithm="viterbi")
    predicted_state_names = [STATES[state] for state in estimated_states]

    plot_parameters(transition, emission)
    print(f"HMM executed successfully using {len(df)} records.")
    print(f"Saved matrix visualization: {OUTPUT_PATH.name}")
    print(
        "Viterbi demonstration [0, 1, 2] -> "
        f"{' -> '.join(predicted_state_names)}"
    )
    print(
        "Probability checks: "
        f"pi={initial.sum():.2f}, "
        f"A rows={np.allclose(transition.sum(axis=1), 1.0)}, "
        f"B rows={np.allclose(emission.sum(axis=1), 1.0)}"
    )


if __name__ == "__main__":
    main()
