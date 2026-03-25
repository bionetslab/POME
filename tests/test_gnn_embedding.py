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
