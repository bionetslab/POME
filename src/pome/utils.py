import torch
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import zscore
import numpy as np

def link_similarity(embeddings, edge_index, decoder_model=None):
    return decoder_model(embeddings, edge_index)


def compute_roc(graph_data, neg_edges_per_pair, node_embeddings, decoder_model):

    device = node_embeddings.device
    pos_edge_index = graph_data.unique_edges.to(device)
    # Sample negative edges running between nodes of group 0 and 1.
    neg_edge_index = torch.cat(neg_edges_per_pair, dim=1).to(device)

    # Compute link probabilities using dot-product.
    pos_scores = link_similarity(node_embeddings, pos_edge_index, decoder_model=decoder_model)
    neg_scores = link_similarity(node_embeddings, neg_edge_index, decoder_model=decoder_model)
    
    # Merge scores and labels for ROC computation using decoder-based similarity.
    y_true = torch.cat([torch.ones(pos_scores.size(0)), torch.zeros(neg_scores.size(0))]).detach()
    y_pred = torch.cat([pos_scores, neg_scores]).detach()
    assert np.all(y_pred.numpy() >= 0)
    assert np.all(y_true.numpy() >= 0)

    roc_auc = roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
    avg_precision = average_precision_score(y_true.numpy(), y_pred.numpy())

    return roc_auc, avg_precision


def effective_rank(Z, eps: float = 1e-7):
    """RankMe effective rank (Garrido et al. 2023) of an embedding matrix.

    The entropy of the normalized singular-value spectrum, exp(-sum p_k log p_k)
    with p_k = sigma_k / sum(sigma) + eps -- a label-free measure of how many
    dimensions the representation effectively uses (bounded by min(n_rows, dim)).
    Non-finite rows are dropped; returns nan if fewer than two usable rows remain.

    Args:
        Z: 2-D array/tensor (n_rows x dim).
        eps (float): numerical floor for the singular-value probabilities.

    Returns:
        float: the effective rank (>= 1), or nan.
    """
    Z = torch.as_tensor(Z, dtype=torch.float32)
    Z = Z[torch.isfinite(Z).all(dim=1)]
    if Z.shape[0] < 2:
        return float("nan")
    sigma = torch.linalg.svdvals(Z)
    p = sigma / (sigma.sum() + eps) + eps
    return float(torch.exp(-(p * torch.log(p)).sum()))


def matched_overfit_index(train_Z, val_Z, n_draws: int = 10, generator=None):
    """Sample-matched, ceiling-normalized train-vs-val effective-rank gap.

    effective_rank is capped/biased by row count, so a raw
    effective_rank(train) - effective_rank(val) is confounded when the two matrices
    have different numbers of rows. Here both are reduced to the same
    n = min(n_train, n_val) rows (the larger one subsampled, averaged over
    `n_draws` draws) so the cap and finite-sample bias match, then the gap is
    normalized by the shared ceiling min(n, dim):

        (effective_rank(train@n) - effective_rank(val@n)) / min(n, dim)

    The residual reflects genuine geometric divergence (overfitting onset) rather
    than the sample-count asymmetry. Both inputs must live in the same latent
    space. Returns nan if fewer than two usable rows.

    Args:
        train_Z / val_Z: 2-D tensors (n_rows x dim) in the same latent space.
        n_draws (int): subsampling draws for the larger matrix.
        generator (torch.Generator): CPU RNG for reproducible subsampling.

    Returns:
        float: the normalized matched gap, or nan.
    """
    train_Z = torch.as_tensor(train_Z, dtype=torch.float32).detach().cpu()
    val_Z = torch.as_tensor(val_Z, dtype=torch.float32).detach().cpu()
    train_Z = train_Z[torch.isfinite(train_Z).all(dim=1)]
    val_Z = val_Z[torch.isfinite(val_Z).all(dim=1)]
    n = min(train_Z.shape[0], val_Z.shape[0])
    d = train_Z.shape[1]
    if n < 2:
        return float("nan")

    def matched(Z):
        if Z.shape[0] == n:
            return effective_rank(Z)
        vals = [effective_rank(Z[torch.randperm(Z.shape[0], generator=generator)[:n]])
                for _ in range(n_draws)]
        return float(np.nanmean(vals))

    rk_train, rk_val = matched(train_Z), matched(val_Z)
    ceiling = min(n, d)
    if not (np.isfinite(rk_train) and np.isfinite(rk_val)) or ceiling < 1:
        return float("nan")
    return (rk_train - rk_val) / ceiling

def get_zscore_bins(K):
    if K == 15:
        return [-np.inf, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, np.inf]
    elif K == 11:
        return [-np.inf, -3.5, -2.5, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.5, 3.5, np.inf]
    elif K == 7:
        return [-np.inf, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, np.inf]
    elif K == 3:
        return [-np.inf, -0.5, 0.5, np.inf]
    else:
        raise ValueError(f"Invalid number of z-score bins: {K}")


def bin_column_with_na_adjusted(column, K, true_missing, keep_nas):
    column = column.copy()  # Avoid modifying the original data
    column = column.astype(float)
    
    # Identify different types of missing values
    true_missing_mask = column == true_missing
    keep_nas_mask = column.isin(keep_nas)
    valid_mask = ~(true_missing_mask | keep_nas_mask | column.isna())

    # Extract non-missing values for binning
    non_na = column[valid_mask]

    # Compute z-scores of non-NA values.
    if len(set(non_na))==1:
        print(f"Warning: cont variable {column.name} contains only one unique value. Setting zscore to 0.")
        non_na_zscores = np.zeros(len(non_na))
    else:
        non_na_zscores = zscore(non_na, nan_policy='raise')
    
    # Perform binning on valid values
    zscore_bins = get_zscore_bins(K)
    
    binned_non_na = pd.cut(non_na_zscores, bins=zscore_bins, labels=False)

    # Create full binned column initialized with NA
    binned_full = pd.Series(pd.NA, index=column.index, dtype="Int64")

    # Assign binned values
    binned_full[valid_mask] = binned_non_na
    
    # Assign separate negative bins to keep_nas values
    for missing_val in keep_nas:
        binned_full[column == missing_val] = missing_val

    # Assign true missing value (-99) to pd.NA
    binned_full[true_missing_mask] = true_missing
    
    return binned_full.astype("float64")

def signed_power_bins(data, n_bins, power=2):
    data = np.asarray(data, dtype=float)

    # Forward transform (power, NOT root!)
    transformed = np.sign(data) * np.abs(data) ** power

    # Create n_bins bins → need n_bins + 1 edges
    t_edges = np.linspace(transformed.min(), transformed.max(), n_bins + 1)

    # Inverse transform
    edges = np.sign(t_edges) * np.abs(t_edges) ** (1 / power)

    # Digitize safely to 0 ... n_bins-1
    # Use internal edges only to avoid -1 / n_bins issues
    bins = np.digitize(data, edges[1:-1], right=True)

    return edges, bins


def bin_column_non_linear(column, K, true_missing, keep_nas):
    column = column.copy()  # Avoid modifying the original data
    column = column.astype(float)
    
    # Identify different types of missing values
    true_missing_mask = column == true_missing
    keep_nas_mask = column.isin(keep_nas)
    valid_mask = ~(true_missing_mask | keep_nas_mask | column.isna())

    # Extract non-missing values for binning
    non_na = column[valid_mask]

    # Compute non-linear bins of non-NA values.
    if len(set(non_na))==1:
        print(f"Warning: cont variable {column.name} contains only one unique value. Setting all bins to 0.")
        binned_non_na = np.full(len(non_na), int((K-1)/2))
    else:
        _, binned_non_na = signed_power_bins(non_na, n_bins=K)

    # Create full binned column initialized with NA
    binned_full = pd.Series(pd.NA, index=column.index, dtype="Int64")

    # Assign binned values
    binned_full[valid_mask] = binned_non_na
    
    # Assign separate negative bins to keep_nas values
    for missing_val in keep_nas:
        binned_full[column == missing_val] = missing_val

    # Assign true missing value (-99) to pd.NA
    binned_full[true_missing_mask] = true_missing
    
    return binned_full.astype("float64")

def repeat_pad_to_max_cols(tensor_list):
    max_cols = max(t.size(1) for t in tensor_list)
    padded_list = []

    for t in tensor_list:
        n_rows, n_cols = t.shape

        if n_cols == max_cols:
            padded_list.append(t)
            continue

        # Repeat columns cyclically to cover max_cols
        n_repeats = (max_cols + n_cols - 1) // n_cols  # ceiling division
        repeated = t.repeat(1, n_repeats)[:, :max_cols]  # repeat along columns

        padded_list.append(repeated)

    return torch.stack(padded_list)  # shape: (B, 2, max_cols)

