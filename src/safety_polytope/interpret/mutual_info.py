import logging
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from omegaconf import DictConfig
from sklearn.metrics import mutual_info_score
from tueplots import bundles

from safety_polytope.data.safety_data import get_hidden_states_dataloader
from safety_polytope.polytope.lm_constraints import PolytopeConstraint

log = logging.getLogger("polytope")


def normalize_violations(matrix):
    # Min-max normalization per edge
    edge_min = matrix.min(axis=1, keepdims=True)
    edge_max = matrix.max(axis=1, keepdims=True)
    normalized = (matrix - edge_min) / (edge_max - edge_min + 1e-8)
    return normalized


def sharpen(matrix, temperature=0.1):
    sharpened = np.exp(matrix / temperature)
    return sharpened / (sharpened.sum(axis=1, keepdims=True) + 1e-8)


def calculate_mutual_information(
    cat_true,
    cat_pred,
    use_raw_violations,
    temperature=0.1,
):
    """
    Calculate the mutual information score between true categories and
    predicted edges or violations, and raise it to a specified power.

    Args:
        cat_true (np.ndarray): Array of shape (n_data, n_categories) with true
            category labels.
        cat_pred (np.ndarray): Array of shape (n_data, n_edges) with
            predicted edge activations or violations.
        use_raw_violations (bool): If True, use a continuous mutual information
            metric. If False, use the discrete mutual information.
        similarity_threshold (float): Threshold for considering mutual
            information vectors as similar.
        power (float): The power to raise the mutual information matrix to.

    Returns:
        np.ndarray: Mutual information matrix of shape (num_categories,
            num_distinct_edges) raised to the specified power.
        list: Indices of the distinct edges.
    """
    num_categories = cat_true.shape[1]
    num_edges = cat_pred.shape[1]
    if use_raw_violations:
        cat_pred = normalize_violations(cat_pred)
        cat_pred = sharpen(cat_pred, temperature=temperature)

    mutual_info_matrix = np.zeros((num_categories, num_edges))

    for i in range(num_categories):
        for j in range(num_edges):
            if use_raw_violations:
                mutual_info_matrix[i, j] = continuous_mutual_info(
                    cat_true[:, i], cat_pred[:, j]
                )
            else:
                # Use the discrete mutual information for binary activations
                mutual_info_matrix[i, j] = mutual_info_score(
                    cat_true[:, i], cat_pred[:, j]
                )

    # Apply Hungarian algorithm for optimal matching
    from scipy.optimize import linear_sum_assignment

    row_ind, col_ind = linear_sum_assignment(-mutual_info_matrix)

    organized_matrix = mutual_info_matrix[:, col_ind]

    organized_matrix = normalize_violations(organized_matrix)

    return organized_matrix, col_ind.tolist()


def continuous_mutual_info(x, y, bins=20):
    """
    Calculate mutual information between a discrete variable and a continuous
    variable.

    Args:
        x (np.ndarray): Discrete variable (e.g., true categories)
        y (np.ndarray): Continuous variable (e.g., raw violations)
        bins (int): Number of bins to use for discretizing the continuous
            variable

    Returns:
        float: Mutual information score
    """
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi


def plot_and_save_heatmap(
    mutual_info_matrix: np.ndarray,
    distinct_edge_indices: list,
    use_raw_violations: bool = False,
):
    """
    Plot and save a heatmap of the mutual information scores with dynamic
    sizing using tueplots for ICML style, with square cells.

    Args:
        mutual_info_matrix (np.ndarray): Mutual information matrix of shape
            (num_categories, num_edges).
        distinct_edge_indices (list): List of distinct edge indices.
        use_raw_violations (bool): Whether to use raw violations instead of
            edge activations for visualization.
    """
    num_categories, num_distinct_edges = mutual_info_matrix.shape
    category_order = [f"{i}" for i in range(num_categories)]
    edge_order = [f"{j}" for j in distinct_edge_indices]

    # Set up ICML style
    plt.rcParams.update(bundles.icml2022())
    plt.rcParams["text.usetex"] = False

    # Increase font sizes
    plt.rcParams["font.size"] = 14  # Base font size
    plt.rcParams["axes.titlesize"] = 20  # Title font size
    plt.rcParams["axes.labelsize"] = 18  # Axis label font size
    plt.rcParams["xtick.labelsize"] = 12  # X-axis tick label size
    plt.rcParams["ytick.labelsize"] = 12  # Y-axis tick label size

    # Calculate figure size to ensure square cells
    cell_size = 0.5  # inches per cell
    fig_width = max(6, num_distinct_edges * cell_size + 2)  # add margin
    fig_height = max(6, num_categories * cell_size + 2)  # add margin

    # Create the figure and axes with constrained_layout
    fig, ax = plt.subplots(
        figsize=(fig_width, fig_height), constrained_layout=True
    )

    # Plot the heatmap without colorbar and with square cells
    sns.heatmap(
        mutual_info_matrix,
        annot=True,
        cmap="viridis",
        fmt=".2f",
        xticklabels=edge_order,
        yticklabels=category_order,
        ax=ax,
        cbar=False,  # Remove colorbar
        square=False,  # Ensure square cells
        annot_kws={"size": 10},  # Adjust annotation font size
    )

    # Set labels and title
    ax.set_title("Mutual Information Heatmap", fontsize=20, pad=20)
    ax.set_xlabel("Learned Facets", fontsize=18, labelpad=10)
    ax.set_ylabel("True Categories", fontsize=18, labelpad=10)

    # Rotate x-axis labels for better readability when there are many edges
    # if num_distinct_edges > 10:
    #     plt.xticks(rotation=45, ha="right")

    # Save the plot
    heatmap_path = os.path.join(os.getcwd(), "mutual_information_heatmap.pdf")
    fig.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Heatmap saved to {heatmap_path}")


def get_class_idx_hidden_state_file(base_path, hs_filename, class_idx):
    hs_filepath = os.path.join(base_path, class_idx, hs_filename)
    hs = torch.load(hs_filepath, weights_only=False)
    return hs


def get_category_true_and_pred(class_idx, model, cfg):
    """
    Generates the true category labels (`y_true`) and predicted edge
    activations or violations (`y_pred`) for a specific category in a dataset
    using a trained polytope model.

    Args:
        class_idx (int): The index of the category for which labels
            will be collected.
        model (torch.nn.Module): The trained polytope model used to generate
            predictions.
        cfg (object): A configuration object containing the dataset paths
            and settings.

    Returns:
        tuple:
            y_true (numpy.ndarray): A binary matrix of shape
            (n_data, num_categories), where each row corresponds to a data
            point and each column corresponds to a category. The values
            match the actual data labels.

            y_pred (numpy.ndarray): A matrix of shape (n_data, num_edges),
            where each row corresponds to a data point and each column
            corresponds to an edge. If use_raw_violations is False, entries
            are binary (0 or 1) indicating edge activation. If True, entries
            are normalized violation scores between 0 and 1.
    """
    data = get_class_idx_hidden_state_file(
        base_path=cfg.dataset.hs_base_path,
        hs_filename=cfg.dataset.hs_filename,
        class_idx=str(class_idx),
    )
    dataloader = get_hidden_states_dataloader(data["test"])
    num_categories = cfg.dataset.num_categories
    num_edges = model.phi.shape[0]
    n_data = len(data["test"]["labels"])

    y_true = np.zeros((n_data, num_categories))
    y_pred = np.zeros((n_data, num_edges))

    start_idx = 0

    for inputs, labels in dataloader:
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(model.device)
        labels = labels.float().to(model.device)
        outputs = model(inputs, label=labels)

        batch_size = inputs.size(0)

        # Fill in the y_true matrix with actual labels
        batch_labels = labels.cpu().numpy()
        y_true[start_idx : start_idx + batch_size, class_idx] = batch_labels

        if cfg.use_raw_violations:
            # Use raw violations
            violations = outputs.violations.cpu().detach().numpy()
            y_pred[start_idx : start_idx + batch_size] = violations
        else:
            # Use binary edge activations
            activate_edge_idx = torch.where(outputs.violations > 0)
            for i in range(len(activate_edge_idx[0])):
                data_idx = start_idx + activate_edge_idx[0][i].item()
                edge_idx = activate_edge_idx[1][i].item()
                y_pred[data_idx, edge_idx] = 1

        start_idx += batch_size

    return y_true, y_pred


def save_cat_data(cat_true, cat_pred, file_path):
    """
    Save cat_true and cat_pred to a PyTorch .pt file.

    Args:
        cat_true (np.ndarray): True category labels
        cat_pred (np.ndarray): Predicted edge activations or violations
        file_path (str): Path to save the .pt file
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Convert numpy arrays to PyTorch tensors
    cat_true_tensor = torch.from_numpy(cat_true)
    cat_pred_tensor = torch.from_numpy(cat_pred)

    # Save tensors
    torch.save(
        {"cat_true": cat_true_tensor, "cat_pred": cat_pred_tensor}, file_path
    )
    log.info(f"Saved cat_true and cat_pred to {file_path}")


def load_cat_data(file_path):
    """
    Load cat_true and cat_pred from a PyTorch .pt file.

    Args:
        file_path (str): Path to the .pt file

    Returns:
        tuple: (cat_true, cat_pred) as numpy arrays
    """
    data = torch.load(file_path, weights_only=False)
    return data["cat_true"].numpy(), data["cat_pred"].numpy()


@hydra.main(
    config_path=f"{os.getcwd()}/exp_configs",
    config_name="interpret_config",
    version_base="1.1",
)
def main(cfg: DictConfig):
    if cfg.cat_data_path and os.path.exists(cfg.cat_data_path):
        log.info(f"Loading cat_true and cat_pred from {cfg.cat_data_path}")
        cat_true, cat_pred = load_cat_data(cfg.cat_data_path)
    else:
        model = PolytopeConstraint(
            model=None,
            tokenizer=None,
            num_phi=cfg.dataset.num_phi,
            train_on_hs=True,
            entropy_assignment=False,
        )
        trained_weights = torch.load(
            cfg.dataset.trained_weights_path, weights_only=False
        )
        model.phi = trained_weights.phi
        model.threshold = trained_weights.threshold
        model.feature_extractor = trained_weights.feature_extractor

        cat_true = []
        cat_pred = []
        log.info("Collecting the true and predicted labels...")
        for i in range(cfg.dataset.num_categories):
            y_true, y_pred = get_category_true_and_pred(i, model, cfg)
            cat_true.append(y_true)
            cat_pred.append(y_pred)

        cat_true = np.concatenate(cat_true, axis=0)
        cat_pred = np.concatenate(cat_pred, axis=0)

        if cfg.save_cat_data_path:
            save_cat_data(cat_true, cat_pred, cfg.save_cat_data_path)

    log.info("Calculating mutual information matrix...")

    mutual_info_matrix, distinct_edge_indices = calculate_mutual_information(
        cat_true,
        cat_pred,
        cfg.use_raw_violations,
        cfg.temperature,
    )

    plot_and_save_heatmap(
        mutual_info_matrix,
        distinct_edge_indices,
        use_raw_violations=cfg.use_raw_violations,
    )


if __name__ == "__main__":
    main()
