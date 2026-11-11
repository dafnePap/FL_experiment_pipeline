import argparse
import os
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "SCAFFOLD": "#24b2d1",
    "FedAvg": "#2ca02c",
    "FedProx": "#1b0db5",
    "FedAdam": "#ff520e",
    "FDAOPT": "#b338a5",
}


def load_results(results_dir):
    results_dir = Path(results_dir)
    files = sorted(results_dir.glob("*.pkl"))

    loaded = {}

    for path in files:
        with open(path, "rb") as f:
            data = pickle.load(f)

        algorithm_name = path.stem.split("_")[0]
        loaded[algorithm_name] = data

    return loaded


def plot_metric(results_dir, metric="accuracy"):
    results_dir = Path(results_dir)
    loaded = load_results(results_dir)

    if not loaded:
        raise ValueError(f"No .pkl files found in {results_dir}")

    plt.figure(figsize=(7, 4))
    max_round = 0

    for algorithm_name, data in loaded.items():
        rounds = data["rounds"]
        values = data[metric]
        max_round = max(max_round, max(rounds))

        plt.plot(
            rounds,
            values,
            label=algorithm_name,
            color=COLORS.get(algorithm_name),
            linewidth=2,
            marker="o",
        )

    plt.xlabel("Communication Round")

    if metric == "accuracy":
        plt.ylabel("Accuracy (%)")
        plt.title("Accuracy vs Communication Round")
    elif metric == "loss":
        plt.ylabel("Loss")
        plt.title("Loss vs Communication Round")

    plt.xticks(range(2, max_round + 1, 2))
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    output = plots_dir / f"{metric}.png"

    if output.exists():
        output.unlink()

    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()

    return output


def generate_all_plots(results_dir):
    accuracy_path = plot_metric(results_dir, "accuracy")
    loss_path = plot_metric(results_dir, "loss")
    return {
        "accuracy": str(accuracy_path),
        "loss": str(loss_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()

    outputs = generate_all_plots(args.results_dir)
    print("Generated plots:")
    for name, path in outputs.items():
        print(f"{name}: {path}")