import logging
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.decomposition import PCA

from safety_polytope.polytope.lm_constraints import PolytopeConstraint


log = logging.getLogger("visualize_violation")


def compute_violation_labels(
    hidden_states: torch.Tensor, model: PolytopeConstraint, batch_size: int = 2048
) -> np.ndarray:
    """
    Compute a boolean violation label per sample.
    A sample is considered negative (violation=True) if any edge has positive violation.
    """
    model.eval()
    labels = []
    with torch.no_grad():
        for start in range(0, hidden_states.shape[0], batch_size):
            end = min(start + batch_size, hidden_states.shape[0])
            batch = hidden_states[start:end].to(model.device)
            print(batch.shape)
            print(model)
            outputs = model(batch, label=None)
            # violations shape: [batch, num_edges]
            batch_viol = outputs.violations
            # violation if any edge > 0
            batch_has_violation = (batch_viol > 0).any(dim=1).cpu().numpy()
            labels.append(batch_has_violation)
    return np.concatenate(labels, axis=0)


def subsample_indices(
    n: int, subset_size: float, max_points: int | None, seed: int = 42
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n * float(subset_size)))
    if max_points is not None:
        k = min(k, int(max_points))
    k = max(1, min(k, n))
    return np.sort(rng.choice(n, size=k, replace=False))


@hydra.main(
    config_path=f"{os.getcwd()}/exp_configs",
    config_name="visualize_violation_config",
    version_base="1.1",
)
def main(cfg: DictConfig):
    # Load hidden states data
    assert (
        cfg.dataset.hidden_states_path is not None
        and cfg.dataset.hidden_states_path != "None"
    ), "dataset.hidden_states_path must be set"
    log.info(f"Loading hidden states from {cfg.dataset.hidden_states_path}")
    hs_data = torch.load(cfg.dataset.hidden_states_path, weights_only=False)
    split = cfg.split
    assert split in hs_data, f"Split '{split}' not found in hidden states file"
    X: torch.Tensor = hs_data[split]["hidden_states"]

    # Optionally subsample for speed/clarity
    idx = subsample_indices(X.shape[0], cfg.subset_size, cfg.max_points, seed=cfg.seed)
    X = X[idx]

    # Load trained polytope weights
    assert (
        cfg.dataset.trained_weights_path is not None
        and cfg.dataset.trained_weights_path != "None"
    ), "dataset.trained_weights_path must be set"
    log.info(f"Loading trained weights from {cfg.dataset.trained_weights_path}")
    trained = torch.load(cfg.dataset.trained_weights_path, weights_only=False)

    # Build a constraint model that operates directly on hidden states
    model = PolytopeConstraint(
        model=None,
        tokenizer=None,
        num_phi=cfg.dataset.num_phi,
        train_on_hs=True,
    )
    model.phi = trained.phi.to(model.device)
    model.threshold = trained.threshold.to(model.device)
    model.feature_extractor = trained.feature_extractor.to(model.device)

    # Compute per-sample violation labels (True if violation present)
    has_violation = compute_violation_labels(X, model)

    # Prepare PCA on raw hidden states
    X_np = X.detach().cpu().numpy()
    pca = PCA(n_components=2, random_state=cfg.seed)
    X_proj = pca.fit_transform(X_np)

    # Map labels to colors: violation (negative) = red, no violation (positive) = blue
    colors = np.where(has_violation, "#d62728", "#1f77b4")
    labels_text = np.where(has_violation, "negative", "positive")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(
        X_proj[:, 0], X_proj[:, 1], c=colors, s=10, alpha=0.7, edgecolors="none"
    )
    # Create a manual legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="negative (violation)",
            markerfacecolor="#d62728",
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="positive (no violation)",
            markerfacecolor="#1f77b4",
            markersize=8,
        ),
    ]
    plt.legend(handles=legend_elements, loc="best")
    plt.title("Hidden states PCA (top-2) with violation labels")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()

    out_path = os.path.join(os.getcwd(), "violation_pca_scatter.png")
    plt.savefig(out_path, dpi=200)
    plt.close()

    frac_neg = has_violation.mean()
    log.info(
        f"Plotted {X.shape[0]} points. Negative fraction (violation): {frac_neg:.3f}"
    )
    log.info(f"Saved PCA scatter to {out_path}")


if __name__ == "__main__":
    main()
