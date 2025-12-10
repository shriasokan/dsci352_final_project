import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_DIR/ "outputs"

LEADERBOARD_PATH = OUTPUTS_DIR / "model_leaderboard_telco.json"
GRADIENTS_PATH = OUTPUTS_DIR / "keras_gradients.json"

def plot_gradients(df_grads: pd.DataFrame, save_path: Path):
    epochs = df_grads["epoch"]
    mean_vals = df_grads["grad_abs_mean"]
    max_vals = df_grads["grad_abs_max"]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, mean_vals, marker="o", label="Mean Gradient")
    plt.plot(epochs, max_vals, marker="x", linestyle="--", label="Max Gradient")
    plt.xlabel("Epoch")
    plt.ylabel("Gradient Magnitude")
    plt.title("Keras Gradient Magnitude over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def save_model_table_png(df_leaderboard: pd.DataFrame, save_path: Path):
    cols = ["name", "type", "auc", "accuracy"]
    df = df_leaderboard[cols].copy()
    df["auc"] = df["auc"].map(lambda x: f"{x:.4f}")
    df["accuracy"] = df["accuracy"].map(lambda x: f"{x:.4f}")

    fig, ax = plt.subplots(figsize=(8, 2 + 0.4 * len(df)))
    ax.axis("off")
    
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    plt.title("Model Validation AUC and Accuracy", pad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(LEADERBOARD_PATH, "r") as f:
        leaderboard_data = json.load(f)
    df_leaderboard = pd.DataFrame(leaderboard_data)

    cols = ["name", "type", "auc", "accuracy"]

    table_png_path = OUTPUTS_DIR / "model_leaderboard_table.png"
    save_model_table_png(df_leaderboard, table_png_path)

    with open(GRADIENTS_PATH, "r") as f:
        gradients_data = json.load(f)
    df_gradients = pd.DataFrame(gradients_data)

    gradients_png_path = OUTPUTS_DIR / "keras_gradients_plot.png"
    plot_gradients(df_gradients, gradients_png_path)

if __name__ == "__main__":
    main()