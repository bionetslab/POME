from pome.gnn_embedding import Embedder, make_deterministic
from pome.utils import effective_rank, matched_overfit_index
import math
import pandas as pd
import torch
import os
import numpy as np
import pytest
import joblib

example_df = pd.read_csv("example.csv", index_col=0)
NA_ENCODING = -99.0
DIMENSION = 16
DEVICE = "cpu"
NUM_SAMPLES = len(example_df.columns)-1
NUM_VARS = len(example_df)
CONT_VARS = [v for v in example_df.index if example_df.loc[v, "type"] in ("numerical", "cont")]
SAMPLE_COLS = [c for c in example_df.columns if c != "type"]

def test_embedder_init():
    
    embedder = Embedder(epochs=100, 
                        na_encoding=NA_ENCODING, 
                        embedding_dimension=DIMENSION,
                        device=DEVICE,
                        enable_imputation=True)
    
    assert embedder.epochs == 100
    assert embedder.device == "cpu"
    assert embedder.bins_per_continuous == 15
    assert embedder.non_informative_na == NA_ENCODING
    assert embedder.embedding_dimension == DIMENSION
    
def test_embedder_fit():
    embedder = Embedder(epochs=100, 
                        na_encoding=NA_ENCODING, 
                        embedding_dimension=DIMENSION,
                        device=DEVICE)
    embedder.fit(example_df)
    
    assert embedder._ap > 0
    assert isinstance(embedder._fitted_decoder, torch.nn.Module)
    assert embedder.return_ap_score() == embedder._ap
    assert isinstance(embedder.decision_function(example_df), np.ndarray)
    
def test_embedding_extraction():
    
    embedder = Embedder(epochs=50, 
                        na_encoding=NA_ENCODING, 
                        embedding_dimension=DIMENSION,
                        device=DEVICE)
    embedder.fit(example_df)
    sample_embeddings, var_embeddings, _ , _ = embedder.get_embeddings()
    
    assert isinstance(sample_embeddings, pd.DataFrame)
    assert len(sample_embeddings.columns)==DIMENSION
    assert len(sample_embeddings.index)==NUM_SAMPLES
    assert len(var_embeddings.columns)==DIMENSION
    assert len(var_embeddings.index)==NUM_VARS
    
def test_imputation():
    
    embedder = Embedder(epochs=50, 
                        na_encoding=NA_ENCODING, 
                        embedding_dimension=DIMENSION,
                        device=DEVICE,
                        enable_imputation=True)
    embedder.fit(example_df)
    imputed_df = embedder.impute_all(na_value=NA_ENCODING)

    assert len(imputed_df.columns)==len(example_df.columns)-1
    assert len(imputed_df.index)==len(example_df.index)
    assert (imputed_df==NA_ENCODING).sum().sum()==0
    
def test_deterministic():
    seed = 42
    make_deterministic(seed)
    assert os.environ["PYTHONHASHSEED"] == str(seed)
    
def test_epoch_checkpoints():
    embedder = Embedder(epochs=50, 
                        na_encoding=NA_ENCODING, 
                        embedding_dimension=DIMENSION,
                        device=DEVICE,
                        enable_imputation=True,
                        file_name="test_checkpoint",
                        output_path=".",
                        epoch_checkpoints=40
                        )
    embedder.fit(example_df)
    assert any(f.endswith('.joblib') for f in os.listdir('.'))

def test_epoch_callback_snapshots():
    # epoch_callback fires once per epoch with a 1-indexed count and the live
    # model; a cloned state_dict reloaded into the final model must reproduce that
    # epoch's embeddings, and an earlier snapshot must differ.
    seen = []
    snaps = {}
    def cb(epoch, autoencoder):
        seen.append(epoch)
        if epoch in (25, 50):
            snaps[epoch] = {k: v.detach().clone()
                            for k, v in autoencoder.state_dict().items()}

    make_deterministic(42)
    embedder = Embedder(epochs=50, na_encoding=NA_ENCODING,
                        embedding_dimension=DIMENSION, device=DEVICE,
                        epoch_callback=cb)
    embedder.fit(example_df)

    assert seen == list(range(1, 51))       # every epoch, 1-indexed, up to `epochs`
    assert set(snaps) == {25, 50}

    edge_index = embedder._graph_data.edge_index
    with torch.no_grad():
        final_nodes = embedder.model.get_embeddings(edge_index)[0]  # epoch-50 weights
        embedder.model.load_state_dict(snaps[50])
        nodes_50 = embedder.model.get_embeddings(edge_index)[0]
        embedder.model.load_state_dict(snaps[25])
        nodes_25 = embedder.model.get_embeddings(edge_index)[0]
    assert torch.allclose(nodes_50, final_nodes)        # last snapshot == final
    assert not torch.allclose(nodes_25, final_nodes)    # earlier snapshot differs

def test_epoch_callback_does_not_perturb_training():
    # The callback only observes; enabling it must not change the training result.
    def fit(cb):
        make_deterministic(42)
        e = Embedder(epochs=30, na_encoding=NA_ENCODING,
                     embedding_dimension=DIMENSION, device=DEVICE,
                     epoch_callback=cb)
        e.fit(example_df)
        return e.get_embeddings()[0].to_numpy()

    without = fit(None)
    with_cb = fit(lambda epoch, autoencoder: None)
    assert np.allclose(without, with_cb)

def test_nonlinear_discretization():
    
    embedder = Embedder(epochs=50, 
                        na_encoding=NA_ENCODING, 
                        embedding_dimension=DIMENSION,
                        device=DEVICE,
                        discretization_type="nonlinear")
    embedder.fit(example_df)
    assert embedder.return_ap_score() > 0
    
def test_bins_parameter():
    embedder = Embedder(epochs=10, 
                        na_encoding=NA_ENCODING, 
                        embedding_dimension=DIMENSION,
                        device=DEVICE,
                        discretization_type="z",
                        bins_per_continuous=11)
    embedder.fit(example_df)
    
    embedder = Embedder(epochs=10, 
                        na_encoding=NA_ENCODING, 
                        embedding_dimension=DIMENSION,
                        device=DEVICE,
                        discretization_type="z",
                        bins_per_continuous=7)
    embedder.fit(example_df)
    
    embedder = Embedder(epochs=50, 
                        na_encoding=NA_ENCODING, 
                        embedding_dimension=DIMENSION,
                        device=DEVICE,
                        discretization_type="z",
                        bins_per_continuous=3)
    embedder.fit(example_df)
    
    embedder = Embedder(epochs=50, 
                        na_encoding=NA_ENCODING, 
                        embedding_dimension=DIMENSION,
                        device=DEVICE,
                        discretization_type="z",
                        bins_per_continuous=2)
    with pytest.raises(ValueError):
            embedder.fit(example_df)
            
def test_embedding_extraction_failure():
    embedder = Embedder(epochs=50, 
                        na_encoding=NA_ENCODING, 
                        embedding_dimension=DIMENSION,
                        device=DEVICE,
                        )
    with pytest.raises(ValueError):
            embedder.get_embeddings()
            
def test_imputation_failure():
    embedder = Embedder(epochs=50, 
                        na_encoding=NA_ENCODING, 
                        embedding_dimension=DIMENSION,
                        device=DEVICE,
                        enable_imputation=False)
    embedder.fit(example_df)
    with pytest.raises(ValueError):
        imputed_df = embedder.impute_all(na_value=NA_ENCODING)
        
def test_layer_failure():
    embedder = Embedder(epochs=50, 
                        na_encoding=NA_ENCODING, 
                        embedding_dimension=DIMENSION,
                        device=DEVICE,
                        enable_imputation=False,
                        layer_type="rubbish")
    with pytest.raises(ValueError):
        embedder.fit(example_df)
        
def test_discretization_failure():
    embedder = Embedder(epochs=50,
                        na_encoding=NA_ENCODING,
                        embedding_dimension=DIMENSION,
                        device=DEVICE,
                        enable_imputation=False,
                        discretization_type="rubbish")
    with pytest.raises(ValueError):
        embedder.fit(example_df)

def _fit_embedder(epochs=50):
    """Helper: fit on the full dataset, return (embedder, last_sample_col)."""
    TYPE_COL = "type"
    sample_cols = [c for c in example_df.columns if c != TYPE_COL]
    embedder = Embedder(epochs=epochs,
                        na_encoding=NA_ENCODING,
                        embedding_dimension=DIMENSION,
                        device=DEVICE)
    embedder.fit(example_df)
    return embedder, sample_cols[-1]

def test_transform_new_samples():
    # Fit on full dataset; all categorical values are guaranteed to be seen.
    TYPE_COL = "type"
    sample_cols = [c for c in example_df.columns if c != TYPE_COL]
    embedder, _ = _fit_embedder(epochs=50)
    new_df = example_df[sample_cols[-2:] + [TYPE_COL]]

    new_embeddings = embedder.transform(new_df)

    assert isinstance(new_embeddings, pd.DataFrame)
    assert new_embeddings.shape == (2, DIMENSION)
    assert list(new_embeddings.index) == sample_cols[-2:]

def test_transform_before_fit():
    embedder = Embedder(epochs=10,
                        na_encoding=NA_ENCODING,
                        embedding_dimension=DIMENSION,
                        device=DEVICE)
    with pytest.raises(ValueError):
        embedder.transform(example_df)

def test_transform_variable_mismatch():
    embedder, held_out = _fit_embedder(epochs=10)

    new_df = example_df[[held_out, "type"]].drop(index="cont.4")
    with pytest.raises(ValueError, match="Variable mismatch"):
        embedder.transform(new_df)

def test_transform_unseen_category():
    embedder, held_out = _fit_embedder(epochs=10)

    # Inject a categorical value never seen during training (valid cats are 0–3).
    # The unseen value is treated as missing: no edge is added for it, but
    # transform() succeeds using the remaining observed features.
    new_df = example_df[[held_out, "type"]].copy()
    new_df.loc["cat", held_out] = 99.0
    result = embedder.transform(new_df)
    assert result.shape == (1, embedder.embedding_dimension)

def test_transform_empty_bin_remapping():
    embedder, held_out = _fit_embedder(epochs=10)

    # An extreme value lands in the highest z-score bin, which is empty with only
    # 10 training samples (|z| > 3.5 requires an outlier beyond 3.5 standard deviations).
    extreme_bin = embedder._get_bin_id("cont", 1000.0)
    assert f"cont={float(extreme_bin)}" not in embedder._value_node_dict

    remapped = embedder._remap_to_populated_bin("cont", extreme_bin)
    assert remapped is not None
    assert f"cont={float(remapped)}" in embedder._value_node_dict

    # transform() must succeed via remapping rather than raising or dropping.
    new_df = example_df[[held_out, "type"]].copy()
    new_df.loc["cont", held_out] = 1000.0
    new_embeddings = embedder.transform(new_df)
    assert new_embeddings.shape == (1, DIMENSION)

def test_impute_sample_failure():
    embedder = Embedder(epochs=10,
                        na_encoding=NA_ENCODING,
                        embedding_dimension=DIMENSION,
                        device=DEVICE,
                        enable_imputation=False)
    embedder.fit(example_df)
    with pytest.raises(ValueError):
        embedder.impute_sample("sample0", NA_ENCODING)

def test_transform_nonlinear_discretization():
    TYPE_COL = "type"
    sample_cols = [c for c in example_df.columns if c != TYPE_COL]
    embedder = Embedder(epochs=50,
                        na_encoding=NA_ENCODING,
                        embedding_dimension=DIMENSION,
                        device=DEVICE,
                        discretization_type="nonlinear")
    embedder.fit(example_df)
    new_df = example_df[[sample_cols[-1], TYPE_COL]]
    new_embeddings = embedder.transform(new_df)
    assert new_embeddings.shape == (1, DIMENSION)

def test_transform_extra_variable():
    embedder, held_out = _fit_embedder(epochs=10)
    new_df = example_df[[held_out, "type"]].copy()
    new_df.loc["extra_var"] = [0.0, "numerical"]
    with pytest.raises(ValueError, match="Variable mismatch"):
        embedder.transform(new_df)

def test_transform_nan_in_input():
    embedder, held_out = _fit_embedder(epochs=10)
    new_df = example_df[[held_out, "type"]].copy()
    new_df.loc["cont", held_out] = np.nan
    with pytest.raises(ValueError, match="Actual NA entry"):
        embedder.transform(new_df)

def test_transform_all_na_input():
    embedder, held_out = _fit_embedder(epochs=10)
    new_df = example_df[[held_out, "type"]].copy()
    new_df.loc[:, held_out] = NA_ENCODING
    with pytest.raises(ValueError, match="No valid edges"):
        embedder.transform(new_df)

def test_transform_one_edgeless_sample_raises():
    # Even a single edgeless sample (all values missing) among otherwise valid samples
    # must raise: it carries no signal for the encoder.
    TYPE_COL = "type"
    sample_cols = [c for c in example_df.columns if c != TYPE_COL]
    embedder, _ = _fit_embedder(epochs=10)
    new_df = example_df[sample_cols[:2] + [TYPE_COL]].copy()
    new_df.loc[:, sample_cols[0]] = NA_ENCODING  # first sample edgeless, second valid
    with pytest.raises(ValueError, match="No valid edges"):
        embedder.transform(new_df)

def test_constant_continuous_variable_z():
    # A continuous variable with a single unique value triggers the zscore constant-var
    # fallback in both data_to_graph (line 503) and bin_column_with_na_adjusted (utils 62-63).
    const_df = example_df.copy()
    sample_cols = [c for c in const_df.columns if c != "type"]
    const_df.loc["cont", sample_cols] = 0.5
    embedder = Embedder(epochs=10,
                        na_encoding=NA_ENCODING,
                        embedding_dimension=DIMENSION,
                        device=DEVICE,
                        discretization_type="z")
    embedder.fit(const_df)
    assert embedder._ap >= 0

def test_constant_continuous_variable_nonlinear():
    # Same scenario with nonlinear discretization covers lines 508-509 and utils 120-121.
    const_df = example_df.copy()
    sample_cols = [c for c in const_df.columns if c != "type"]
    const_df.loc["cont", sample_cols] = 0.5
    embedder = Embedder(epochs=10,
                        na_encoding=NA_ENCODING,
                        embedding_dimension=DIMENSION,
                        device=DEVICE,
                        discretization_type="nonlinear")
    embedder.fit(const_df)
    assert embedder._ap >= 0

def test_informative_na_z():
    # Informative NAs get their own embedding slot (gnn line 572, utils line 80).
    INFORMATIVE_NA = -88.0
    inf_df = example_df.copy()
    sample_cols = [c for c in inf_df.columns if c != "type"]
    inf_df.loc["cont", sample_cols[0]] = INFORMATIVE_NA
    embedder = Embedder(epochs=10,
                        na_encoding=NA_ENCODING,
                        embedding_dimension=DIMENSION,
                        device=DEVICE,
                        informative_nas=[INFORMATIVE_NA],
                        discretization_type="z")
    embedder.fit(inf_df)
    assert embedder._ap >= 0

def _regression_embedder(discretization_type="z", epochs=50):
    return Embedder(epochs=epochs,
                    na_encoding=NA_ENCODING,
                    embedding_dimension=DIMENSION,
                    device=DEVICE,
                    enable_imputation=True,
                    discretization_type=discretization_type)

def test_regression_imputation_z():
    embedder = _regression_embedder("z")
    embedder.fit(example_df)
    imputed_df = embedder.impute_all(na_value=NA_ENCODING)

    assert imputed_df.shape == (NUM_VARS, NUM_SAMPLES)
    assert (imputed_df == NA_ENCODING).sum().sum() == 0
    assert isinstance(embedder._value_regressor, torch.nn.Module)

def test_regression_imputation_nonlinear():
    embedder = _regression_embedder("nonlinear")
    embedder.fit(example_df)
    imputed_df = embedder.impute_all(na_value=NA_ENCODING)

    assert (imputed_df == NA_ENCODING).sum().sum() == 0
    # Every continuous variable observed in training gets standardization stats.
    assert all(v in embedder._reg_target_stats for v in CONT_VARS)

def test_imputation_fits_regressor_by_default():
    # Enabling imputation with continuous variables always trains the regression head.
    embedder = Embedder(epochs=50,
                        na_encoding=NA_ENCODING,
                        embedding_dimension=DIMENSION,
                        device=DEVICE,
                        enable_imputation=True)
    embedder.fit(example_df)
    assert isinstance(embedder._value_regressor, torch.nn.Module)
    imputed_df = embedder.impute_all(na_value=NA_ENCODING)
    assert (imputed_df == NA_ENCODING).sum().sum() == 0

def test_regression_output_finite():
    embedder = _regression_embedder("z")
    embedder.fit(example_df)
    imputed_df = embedder.impute_all(na_value=NA_ENCODING)
    cont_cells = imputed_df.loc[CONT_VARS, SAMPLE_COLS].to_numpy(dtype=float)
    assert np.isfinite(cont_cells).all()

def test_regression_without_imputation():
    embedder = Embedder(epochs=50,
                        na_encoding=NA_ENCODING,
                        embedding_dimension=DIMENSION,
                        device=DEVICE,
                        enable_imputation=False)
    embedder.fit(example_df)
    assert not hasattr(embedder, "_value_regressor")
    with pytest.raises(ValueError):
        embedder.impute_all(na_value=NA_ENCODING)

def test_regressor_persistence(tmp_path):
    make_deterministic(42)
    embedder = _regression_embedder("z")
    embedder.fit(example_df)
    before = embedder.impute_all(na_value=NA_ENCODING)

    path = tmp_path / "reg_embedder.joblib"
    joblib.dump(embedder, path)
    reloaded = joblib.load(path)
    after = reloaded.impute_all(na_value=NA_ENCODING)

    assert isinstance(reloaded._value_regressor, torch.nn.Module)
    pd.testing.assert_frame_equal(before, after)

def test_informative_na_nonlinear():
    # Same with nonlinear discretization covers utils line 133.
    INFORMATIVE_NA = -88.0
    inf_df = example_df.copy()
    sample_cols = [c for c in inf_df.columns if c != "type"]
    inf_df.loc["cont", sample_cols[0]] = INFORMATIVE_NA
    embedder = Embedder(epochs=10,
                        na_encoding=NA_ENCODING,
                        embedding_dimension=DIMENSION,
                        device=DEVICE,
                        informative_nas=[INFORMATIVE_NA],
                        discretization_type="nonlinear")
    embedder.fit(inf_df)
    assert embedder._ap >= 0


# --- Inductive CV-based epoch tuning -------------------------------------------------

def test_inductive_epoch_tuning():
    # With inductive=True, fit() first runs a sample-holdout CV to pick the epoch count, then
    # trains on all samples for that many epochs. The epoch cap must be restored afterwards.
    TYPE_COL = "type"
    sample_cols = [c for c in example_df.columns if c != TYPE_COL]
    make_deterministic(42)
    embedder = Embedder(inductive=True,
                        epochs=60,
                        cv_folds=3,
                        cv_eval_every=10,
                        cv_patience=2,
                        na_encoding=NA_ENCODING,
                        embedding_dimension=DIMENSION,
                        device=DEVICE)
    embedder.fit(example_df)

    assert hasattr(embedder, "_optimal_epochs")
    assert 0 < embedder._optimal_epochs <= 60
    assert embedder._optimal_epochs % embedder.cv_eval_every == 0
    assert embedder._max_epochs == 60
    assert embedder.epochs == 60  # cap restored so the estimator stays re-fittable

    # Fitted embeddings and inductive transform still work.
    emb_df, var_df, _, _ = embedder.get_embeddings()
    assert emb_df.shape == (NUM_SAMPLES, DIMENSION)
    new_embeddings = embedder.transform(example_df[sample_cols[-2:] + [TYPE_COL]])
    assert new_embeddings.shape == (2, DIMENSION)


def test_inductive_epoch_tuning_deterministic():
    # Same seed + cv_seed => identical selected epoch count.
    results = []
    for _ in range(2):
        make_deterministic(42)
        embedder = Embedder(inductive=True,
                            epochs=40,
                            cv_folds=3,
                            cv_eval_every=10,
                            cv_patience=2,
                            na_encoding=NA_ENCODING,
                            embedding_dimension=DIMENSION,
                            device=DEVICE)
        embedder.fit(example_df)
        results.append(embedder._optimal_epochs)
    assert results[0] == results[1]


def test_inductive_off_unchanged():
    # Default (inductive=False) must not run CV nor set the tuning attributes.
    embedder = Embedder(epochs=20,
                        na_encoding=NA_ENCODING,
                        embedding_dimension=DIMENSION,
                        device=DEVICE)
    embedder.fit(example_df)
    assert not hasattr(embedder, "_optimal_epochs")
    assert embedder.epochs == 20


def test_inductive_too_few_folds_fallback():
    # More folds than samples => CV is infeasible; fit() falls back to the fixed epoch count.
    embedder = Embedder(inductive=True,
                        epochs=15,
                        cv_folds=NUM_SAMPLES + 5,
                        na_encoding=NA_ENCODING,
                        embedding_dimension=DIMENSION,
                        device=DEVICE)
    embedder.fit(example_df)
    assert embedder._optimal_epochs == 15
    assert embedder.epochs == 15


# --- Label-free effective-rank metric (matched-gap stopping signal) ------------------

def test_effective_rank():
    # Identity uses all dimensions -> effective rank == n; a rank-1 (collapsed) matrix -> ~1.
    assert abs(effective_rank(torch.eye(8)) - 8.0) < 1e-3
    collapsed = torch.arange(1.0, 9.0).reshape(-1, 1) @ torch.ones(1, 8)  # every row colinear
    assert effective_rank(collapsed) < 1.5
    assert math.isnan(effective_rank(torch.ones(1, 8)))  # < 2 usable rows


def test_matched_overfit_index():
    gen = torch.Generator().manual_seed(0)
    Z = torch.randn(6, 8)
    # Identical matrices -> zero gap.
    assert abs(matched_overfit_index(Z, Z, n_draws=5, generator=gen)) < 1e-6
    # Train richer (full rank) than val (collapsed) -> positive gap.
    full = torch.eye(8)
    collapsed = torch.arange(1.0, 9.0).reshape(-1, 1) @ torch.ones(1, 8)
    assert matched_overfit_index(full, collapsed, n_draws=5, generator=gen) > 0
    # Fewer than two usable rows -> nan.
    assert math.isnan(matched_overfit_index(torch.ones(1, 8), Z, n_draws=5, generator=gen))


def test_inductive_matched_gap():
    # Inductive CV driven by the matched-gap criterion, with explicit tuning params: it runs,
    # selects a valid epoch budget (onset epoch, a multiple of cv_eval_every, or the full cap),
    # restores the cap, and leaves inductive transform working.
    TYPE_COL = "type"
    sample_cols = [c for c in example_df.columns if c != TYPE_COL]
    make_deterministic(42)
    embedder = Embedder(inductive=True,
                        epochs=60,
                        cv_folds=3,
                        cv_eval_every=10,
                        cv_patience=2,
                        overfit_tol=0.05,
                        gap_draws=5,
                        na_encoding=NA_ENCODING,
                        embedding_dimension=DIMENSION,
                        device=DEVICE)
    embedder.fit(example_df)

    assert embedder.overfit_tol == 0.05 and embedder.gap_draws == 5
    assert 0 < embedder._optimal_epochs <= 60
    assert embedder._optimal_epochs % embedder.cv_eval_every == 0
    assert embedder._max_epochs == 60
    assert embedder.epochs == 60  # cap restored

    new_embeddings = embedder.transform(example_df[sample_cols[-2:] + [TYPE_COL]])
    assert new_embeddings.shape == (2, DIMENSION)


# ---------------------------------------------------------------------------------------------
# Identifiable variable representations (non-identifiability of raw e_j / e_c / e_b)
# ---------------------------------------------------------------------------------------------

def _fitted_embedder(seed=0, **kwargs):
    make_deterministic(seed)
    embedder = Embedder(epochs=50, na_encoding=NA_ENCODING, embedding_dimension=DIMENSION,
                        device=DEVICE, **kwargs)
    embedder.fit(example_df)
    return embedder


def _apply_symmetries(embedder, seed=0):
    """Apply exact loss-preserving reparametrizations to the learned encodings in place:
    per-categorical-variable shifts and per-node rescalings, one shared numeric shift,
    per-sample rescaling. Returns the edge index needed to recompute z."""
    rng = np.random.default_rng(seed)
    model = embedder.model
    W = model.node_embeddings.weight.data
    n2e = model.node_to_embeddings.cpu()
    n = len(embedder._sample_node_dict)
    var_rows = dict(zip(embedder._variable_names, model.variable_embeddings_ids.cpu().tolist()))
    value_rows = n2e[n:]
    rand = lambda: torch.tensor(rng.normal(size=W.size(1)), dtype=W.dtype)
    for var, row in var_rows.items():
        if var in embedder._cont_vars:
            continue
        codes = value_rows[value_rows[:, 0] == row, 1]
        a = rand(); W[row] += a; W[codes] -= a                  # categorical shift
        for c in codes:                                         # per-node rescaling of e_j + e_c
            lam = float(np.exp(rng.uniform(np.log(0.2), np.log(5))))
            W[c] = lam * (W[row] + W[c]) - W[row]
    num_rows = [var_rows[v] for v in embedder._variable_names if v in embedder._cont_vars]
    a = rand(); W[num_rows] += a; W[model.bin_embedding_ids.long()] -= a   # shared numeric shift
    W[:n] *= torch.tensor(np.exp(rng.uniform(-1, 1, size=(n, 1))), dtype=W.dtype)
    z, _, _ = model.get_embeddings(embedder._graph_data.edge_index)
    return z


def test_latent_embeddings_invariant_to_encoding_symmetries():
    embedder = _fitted_embedder()
    z_before = embedder._all_embeddings.clone()
    z_after = _apply_symmetries(embedder)
    assert torch.allclose(z_before, z_after, atol=1e-5)


def test_variable_and_bin_representations_invariant():
    embedder = _fitted_embedder()
    _, var_before, bin_before, _ = embedder.get_embeddings()
    _, raw_before, _, _ = embedder.get_embeddings(raw=True)
    concept_before = embedder.get_numeric_concept_embeddings()
    value_before = embedder.get_value_node_embeddings(level="input")
    embedder._all_embeddings = _apply_symmetries(embedder)
    _, var_after, bin_after, _ = embedder.get_embeddings()
    _, raw_after, _, _ = embedder.get_embeddings(raw=True)
    raw_after = pd.DataFrame(embedder.model.node_embeddings.weight.detach()[
        embedder.model.variable_embeddings_ids].numpy(), index=raw_after.index,
        columns=raw_after.columns)
    np.testing.assert_allclose(var_before.to_numpy(), var_after.to_numpy(), atol=1e-5)
    np.testing.assert_allclose(bin_before.to_numpy(), bin_after.to_numpy(), atol=1e-5)
    np.testing.assert_allclose(concept_before.to_numpy(),
                               embedder.get_numeric_concept_embeddings().to_numpy(), atol=1e-5)
    np.testing.assert_allclose(value_before.to_numpy(),
                               embedder.get_value_node_embeddings(level="input").to_numpy(),
                               atol=1e-5)
    # Sanity check that the transform did something: the raw encodings change.
    assert not np.allclose(raw_before.to_numpy(), raw_after.to_numpy(), atol=1e-3)


def test_raw_embeddings_match_learned_encodings():
    embedder = _fitted_embedder()
    _, raw_df, raw_bins, _ = embedder.get_embeddings(raw=True)
    np.testing.assert_allclose(raw_df.to_numpy(), embedder._variable_embeddings.numpy())
    np.testing.assert_allclose(raw_bins.to_numpy(), embedder._bin_embeddings.numpy())


def test_variable_representation_is_value_node_centroid():
    embedder = _fitted_embedder()
    _, var_df, _, _ = embedder.get_embeddings()
    values = embedder.get_value_node_embeddings(level="input")
    for var in var_df.index:
        rows = [label for label in values.index if label.split("=")[0] == var]
        np.testing.assert_allclose(var_df.loc[var].to_numpy(),
                                   values.loc[rows].mean(axis=0).to_numpy(), atol=1e-6)
    norms = embedder.get_variable_centroid_norms()
    np.testing.assert_allclose(norms.to_numpy(), np.linalg.norm(var_df.to_numpy(), axis=1),
                               atol=1e-6)
    assert (norms <= 1 + 1e-6).all()


def test_numeric_concepts_centred_and_scaled():
    embedder = _fitted_embedder()
    concepts = embedder.get_numeric_concept_embeddings()
    assert list(concepts.index) == [v for v in embedder._variable_names if v in CONT_VARS]
    np.testing.assert_allclose(concepts.mean(axis=0).to_numpy(), 0, atol=1e-6)
    assert np.isclose(np.linalg.norm(concepts.to_numpy(), axis=1).mean(), 1.0)


def test_value_node_embeddings_levels():
    embedder = _fitted_embedder()
    latent = embedder.get_value_node_embeddings(level="latent")
    inputs = embedder.get_value_node_embeddings(level="input")
    assert list(latent.index) == list(embedder._value_node_dict.keys())
    assert latent.shape == inputs.shape and latent.shape[1] == DIMENSION
    np.testing.assert_allclose(np.linalg.norm(inputs.to_numpy(), axis=1), 1, atol=1e-5)
    row = embedder._value_node_dict[latent.index[0]]
    np.testing.assert_allclose(latent.iloc[0].to_numpy(), embedder._all_embeddings[row].numpy())
    with pytest.raises(ValueError):
        embedder.get_value_node_embeddings(level="foo")


def _refit_regressor(embedder, seed=0):
    torch.manual_seed(seed)
    embedder._fit_value_regressor()
    return {(v, s): embedder._regress_value(v, s)
            for v in sorted(embedder._reg_target_stats) for s in list(embedder._sample_node_dict)[:5]}


def test_regressor_inputs_invariant_to_encoding_symmetries():
    embedder = _fitted_embedder(enable_imputation=True)
    assert embedder.regressor_inputs == "invariant"
    before = _refit_regressor(embedder)
    samples_before = embedder._sample_encodings.clone()
    vars_before = {k: v.clone() for k, v in embedder._reg_variable_features.items()}
    head = embedder._value_regressor
    _apply_symmetries(embedder)
    _refit_regressor(embedder)                 # recomputes the input features
    assert torch.allclose(samples_before, embedder._sample_encodings, atol=1e-5)
    for k, v in vars_before.items():
        assert torch.allclose(v, embedder._reg_variable_features[k], atol=1e-5)
    # The same trained head gives the same imputations. (Retraining the head is not compared:
    # full-batch L1 training amplifies float-level input differences, for any input choice.)
    embedder._value_regressor = head
    after = {k: embedder._regress_value(*k) for k in before}
    assert before and all(abs(before[k] - after[k]) < 1e-4 for k in before)


def test_raw_regressor_inputs_option():
    make_deterministic(0)
    embedder = Embedder(epochs=50, na_encoding=NA_ENCODING, embedding_dimension=DIMENSION,
                        device=DEVICE, regressor_inputs="raw", enable_imputation=True)
    embedder.fit(example_df)
    k = embedder._variable_names.index(sorted(embedder._reg_variable_features)[0])
    assert torch.equal(embedder._reg_variable_features[embedder._variable_names[k]],
                       embedder._variable_embeddings[k].detach().cpu())
    with pytest.raises(ValueError):
        Embedder(regressor_inputs="other")
