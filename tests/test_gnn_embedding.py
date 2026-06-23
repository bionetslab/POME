from pome.gnn_embedding import Embedder, make_deterministic
import pandas as pd
import torch
import os
import numpy as np
import pytest

example_df = pd.read_csv("example.csv", index_col=0)
NA_ENCODING = -99.0
DIMENSION = 16
DEVICE = "cpu"
NUM_SAMPLES = len(example_df.columns)-1
NUM_VARS = len(example_df)

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
