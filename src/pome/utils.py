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

