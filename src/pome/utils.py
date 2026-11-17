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



def _spectral_components(unique_edges, num_samples, num_nodes, d):
    """Non-trivial singular triplets of the degree-normalized sample x value biadjacency.

    Returns (U, S, V) with at most d components (first, trivial degree component dropped),
    sorted by decreasing singular value, signs fixed so that each component's largest-|.|
    sample entry is positive. Deterministic given the graph.
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import svds

    src, dst = unique_edges.detach().cpu().numpy()
    num_values = num_nodes - num_samples
    A = sp.csr_matrix((np.ones(len(src)), (src, dst - num_samples)),
                      shape=(num_samples, num_values))
    A.data[:] = 1.0  # duplicate edges count once
    deg_s = np.maximum(np.asarray(A.sum(axis=1)).ravel(), 1.0)
    deg_v = np.maximum(np.asarray(A.sum(axis=0)).ravel(), 1.0)
    A_hat = sp.diags(deg_s ** -0.5) @ A @ sp.diags(deg_v ** -0.5)

    k = min(d + 1, min(A_hat.shape) - 1)
    if k < 1:
        return np.zeros((num_samples, 0)), np.zeros(0), np.zeros((num_values, 0))
    U, S, Vt = svds(A_hat, k=k, random_state=0, solver="arpack",
                    v0=np.ones(min(A_hat.shape)))
    order = np.argsort(-S)
    U, S, V = U[:, order][:, 1:], S[order][1:], Vt[order].T[:, 1:]
    signs = np.sign(U[np.abs(U).argmax(axis=0), np.arange(U.shape[1])])
    signs[signs == 0] = 1.0
    return U * signs, S, V * signs


def spectral_sample_embedding(unique_edges, num_samples, num_nodes, d):
    """Unit-length spectral sample coordinates x_i = U_i Sigma^1/2 (at most d components).

    The reference geometry of the output Gram anchor: its Gram matrix X X^T holds the cosine
    similarities of the samples in the spectral embedding of the bipartite sample-value graph.
    No padding, so the result is fully deterministic (also when fewer than d components exist).

    Returns:
        Tensor[num_samples, c] (float32, CPU), c <= d.
    """
    U, S, _ = _spectral_components(unique_edges, num_samples, num_nodes, d)
    X = U * np.sqrt(S)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    X = np.where(n > 0, X / np.where(n > 0, n, 1.0), X)
    return torch.tensor(X, dtype=torch.float32)


def spectral_nystrom_projection(unique_edges, num_samples, num_nodes, d):
    """Per-value-node Nystrom weights for out-of-sample spectral sample coordinates.

    For the degree-normalized biadjacency A_hat = D_s^-1/2 A D_v^-1/2 = U S V^T, a training
    sample's spectral coordinate is x_i = U_i S^1/2 = A_hat_i V S^-1/2. The same map applied to
    a new sample's (training-degree-normalized) row of value nodes N(s) gives

        x_s = deg_s^-1/2 * sum_{v in N(s)} deg_v^-1/2 V_v S^-1/2,

    which reproduces x_i exactly for a training sample. Since the coordinates are unit-normalized
    afterwards, the deg_s factor can be dropped: x_s is proportional to the sum of the returned
    rows P_v = deg_v^-1/2 V_v S^-1/2 over the sample's value nodes.

    Returns:
        ndarray[num_nodes - num_samples, c] (float64), c <= d; row v belongs to value node
        num_samples + v. Uses the same components (and signs) as ``spectral_encoding_table``.
    """
    import scipy.sparse as sp

    U, S, V = _spectral_components(unique_edges, num_samples, num_nodes, d)
    src, dst = unique_edges.detach().cpu().numpy()
    A = sp.csr_matrix((np.ones(len(src)), (src, dst - num_samples)),
                      shape=(num_samples, num_nodes - num_samples))
    A.data[:] = 1.0
    deg_v = np.maximum(np.asarray(A.sum(axis=0)).ravel(), 1.0)
    return V * (S ** -0.5) * (deg_v ** -0.5)[:, None]


def gram_discrepancy(Z, X):
    """Mean squared difference of the Gram matrices, mean_ij (z_i.z_j - x_i.x_j)^2.

    Computed as (||Z^T Z||^2 - 2 ||Z^T X||^2 + ||X^T X||^2) / n^2, which needs only d x d
    products (O(n d^2) time, O(d^2) memory) instead of the n x n Gram matrices. Invariant to
    rotations of Z (and of X).
    """
    n = Z.shape[0]
    zz = (Z.T @ Z).pow(2).sum()
    zx = (Z.T @ X).pow(2).sum()
    xx = (X.T @ X).pow(2).sum()
    return (zz - 2 * zx + xx) / (n * n)


def spectral_encoding_table(unique_edges, num_samples, num_nodes, node_to_embeddings,
                            base_table, als_iters=100):
    """Spectral (bipartite Laplacian-eigenmap / MCA) initialization of POME's encoding table.

    1. Biadjacency A (samples x value nodes) from the positive sample->value edges, degree
       normalized, D_s^-1/2 A D_v^-1/2, truncated SVD; the first (trivial, degree) component is
       dropped. Sample targets x_i = U_i Sigma^1/2, value-node targets t_v = V_v Sigma^1/2, each
       scaled to unit length. Component signs are fixed (largest-|.| sample entry positive), so
       the result is deterministic.
    2. Sample rows (node -> [i, i]) are set to x_i.
    3. Value nodes use normalize(e_a + e_b) (variable row a, value row b). The shared rows are
       fitted to the targets by least squares, sum_v ||e_a(v) + e_b(v) - t_v||^2, with alternating
       means: variable rows start at the mean target of their value nodes, then value rows and
       variable rows are updated in turn. Rows that belong to one variable only (categories) are
       matched exactly; rows shared across variables (numeric bins, informative NAs) get the
       best additive fit.

    If the graph supports fewer than d non-trivial components the remaining coordinates are
    filled from ``base_table`` scaled by 1e-2; rows not used by any node keep ``base_table``.

    Args:
        unique_edges (LongTensor[2, E]): sample node -> value node edges.
        num_samples (int), num_nodes (int): node counts (samples are nodes 0..num_samples-1).
        node_to_embeddings (LongTensor[num_nodes, 2]): table rows summed per node.
        base_table (Tensor[num_rows, d]): current (random) table, used for padding / unused rows.

    Returns:
        Tensor[num_rows, d] (float32, CPU).
    """
    base = base_table.detach().cpu().double()
    num_rows, d = base.shape
    U, S, V = _spectral_components(unique_edges, num_samples, num_nodes, d)
    c = U.shape[1]
    X = np.zeros((num_samples, d))
    T = np.zeros((num_nodes - num_samples, d))
    X[:, :c] = U * np.sqrt(S)
    T[:, :c] = V * np.sqrt(S)
    if c < d:  # pad missing components with small noise from the base table
        n2e = node_to_embeddings.cpu().numpy()
        X[:, c:] = 1e-2 * base.numpy()[n2e[:num_samples, 0], c:]
        T[:, c:] = 1e-2 * (base.numpy()[n2e[num_samples:, 0], c:]
                           + base.numpy()[n2e[num_samples:, 1], c:])

    def unit(M):
        n = np.linalg.norm(M, axis=1, keepdims=True)
        return np.where(n > 0, M / np.where(n > 0, n, 1.0), M)

    X, T = unit(X), unit(T)
    W = base.numpy().copy()
    n2e = node_to_embeddings.cpu().numpy()
    W[n2e[:num_samples, 0]] = X

    a, b = n2e[num_samples:, 0], n2e[num_samples:, 1]
    used_a, inv_a = np.unique(a, return_inverse=True)
    used_b, inv_b = np.unique(b, return_inverse=True)
    cnt_a = np.bincount(inv_a, minlength=len(used_a))[:, None]
    cnt_b = np.bincount(inv_b, minlength=len(used_b))[:, None]

    def group_mean(values, inv, cnt):
        out = np.zeros((cnt.shape[0], d))
        np.add.at(out, inv, values)
        return out / cnt

    E_a = group_mean(T, inv_a, cnt_a)
    E_b = np.zeros((len(used_b), d))
    for _ in range(als_iters):
        E_b = group_mean(T - E_a[inv_a], inv_b, cnt_b)
        E_a_new = group_mean(T - E_b[inv_b], inv_a, cnt_a)
        if np.allclose(E_a_new, E_a, atol=1e-10):
            E_a = E_a_new
            break
        E_a = E_a_new
    W[used_a] = E_a
    W[used_b] = E_b
    return torch.tensor(W, dtype=torch.float32)
