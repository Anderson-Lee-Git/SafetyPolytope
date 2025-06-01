import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tueplots import bundles


def plot_edge_kl_divergences(csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # Read the CSV file
    df = pd.read_csv(csv_path)

    edges = df["Edge"].unique()
    all_words = df["Word"].unique()

    sns.set_theme(style="darkgrid")
    plt.rcParams.update(bundles.icml2022())
    plt.rcParams["text.usetex"] = False

    # Update font sizes
    plt.rcParams["font.size"] = 12  # Base font size
    plt.rcParams["axes.titlesize"] = 12  # Title font size
    plt.rcParams["axes.labelsize"] = 14  # Axis label font size
    plt.rcParams["xtick.labelsize"] = 8  # X tick label size
    plt.rcParams["ytick.labelsize"] = 10  # Y tick label size

    palette = plt.cm.viridis(np.linspace(0, 1, len(all_words)))

    for edge in edges:
        # Create a complete dataset for this edge with all words
        edge_data = []
        for word in all_words:
            # Find KL divergence for this edge-word pair
            kl_value = df[(df["Edge"] == edge) & (df["Word"] == word)][
                "KL Divergence"
            ].values
            # If no value exists, use 0
            kl_value = kl_value[0] if len(kl_value) > 0 else 0.0
            edge_data.append({"Word": word, "KL Divergence": kl_value})

        edge_df = pd.DataFrame(edge_data)
        # Sort by KL Divergence from max to min
        edge_df = edge_df.sort_values("KL Divergence", ascending=False)

        # Create the plot with adjusted figsize
        fig = plt.figure(
            figsize=(
                bundles.icml2022()["figure.figsize"][0] * 1.2,
                bundles.icml2022()["figure.figsize"][1] * 1.2,
            )
        )
        ax = fig.add_subplot(111)

        # Enhanced grid
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)

        # Create horizontal bar plot
        bars = sns.barplot(
            x="KL Divergence",
            y="Word",
            data=edge_df,
            orient="h",
            ax=ax,
            palette=palette,
        )

        ax.set_title(f"KLD for Facet {edge}")

        # Remove x and y axis labels
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Add value labels to the end of each bar with enhanced styling
        for i, v in enumerate(edge_df["KL Divergence"]):
            ax.text(
                v,
                i,
                f" {v:.3f}",
                va="center",
                fontsize=8,
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.7,
                    pad=0.5,
                ),
            )

        # Set x-axis to start at 0 if all values are positive
        if edge_df["KL Divergence"].min() >= 0:
            ax.set_xlim(left=0)

        # Adjust layout
        plt.tight_layout()

        # Save the plot with higher DPI and proper format
        save_path = os.path.join(
            output_dir, f"edge_{int(edge)}_kl_divergence.pdf"
        )
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"Plots have been saved in {output_dir}")


if __name__ == "__main__":
    csv_path = "placeholder"
    output_dir = "notebooks/figures/edge_kl_plots"
    plot_edge_kl_divergences(csv_path, output_dir)
