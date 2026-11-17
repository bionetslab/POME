# POME: Learning partially observed mixed-type data embeddings 
![Tests](https://github.com/bionetslab/POME/actions/workflows/tests.yaml/badge.svg)
![Coverage Status](./coverage.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyPI](https://img.shields.io/pypi/v/pome-py?color=orange)

<p align="center">
  <img width="400" height="410" alt="POME_logo-01" src="https://github.com/user-attachments/assets/67f3d5a0-da8e-4cf0-afec-c12efd4a4b14" />
</p>

POME is a graph-based representation-learning method for heterogeneous datasets that incorporates missingness structures into the computation of low-dimensional sample and variable embeddings. It is applicable to any tabular datasets consisting of both numeric- and categorical-type features, where missing data patterns are supposed to be taken into account.

## Installation
POME is implemented as a Python package and is easily installable from PyPI by running
```
pip install pome-py
```
or locally from this repository by running
```
pip install .
```

## Input format
POME expects input data to be given in the form of a pandas dataframe object, with rows representing variables/features and columns representing samples. Missing data needs to be encoded by a unique numerical value. Furthermore, POME expects one column storing datatypes of the respective variables. An example dataset could have the following structure, with e.g. value -99 encoding missing data:
| | **Sample1** | **Sample2** | **Sample3** | **Type**
|----------|----------|----------|--------|----------|
| **VariableA**   | 0   | 1   | -99 | cat | 
| **VariableB**   | 3.14   | -0.1   | 2.5 | numerical |
| **VariableC**   | 0.3    | 1.2   | -99 | numerical |
| **VariableD**   | 1    | 0   | 2 | cat |

## Minimal working example
POME's core functionality is integrated into its `Embedder` class, which handles input transformation, training and output generation. A typical such workflow looks as follows:
```python
import pandas as pd
from pome import Embedder

if __name__ == "__main__":
    # Load data and set parameters.
    example_df = pd.read_csv("example.csv", index_col=0)
    NA_ENCODING = -99.0
    DIMENSION = 16
    DEVICE = "cpu"
    # Initialize embedding object with parameters.
    embedder = Embedder(epochs=100, 
                        na_encoding=NA_ENCODING, 
                        embedding_dimension=DIMENSION,
                        device=DEVICE,
                        enable_imputation=True)
    # Fit embedding object to dataset.
    embedder.fit(example_df)
    # Output stores low-dimensional embeddings for samples and variables.
    sample_embeddings, variable_embeddings, _ , _ = embedder.get_embeddings()
    print("Computed sample embeddings: \n", sample_embeddings)
    imputed_df = embedder.impute_all(na_value=NA_ENCODING)
    print("Imputed data: \n", imputed_df)
```

## Embedding unseen samples
Once fitted, POME can embed new samples that were not part of the training data using the frozen trained encoder, without any retraining. When new samples are the goal, `inductive=True` additionally replaces the fixed epoch budget by a cross-validated one that is selected to generalize to unseen samples:
```python
from pome import Embedder, make_deterministic

embedder = Embedder(epochs=500, na_encoding=-99.0, embedding_dimension=16, inductive=True)
embedder.fit(train_df)                        # CV-tuned epoch count, stored in _optimal_epochs
new_embeddings = embedder.transform(test_df)  # frozen encoder, no retraining
```

## Parameters

POME's Embedder class allows for the specification of the following parameters:

### Core
- `embedding_dimension : int = 32`: Specifies the number of dimensions of the sample & variable embeddings learned by POME.
- `epochs : int = 500`: Sets the number of epochs that POME is supposed to be trained.
- `device : str = "cpu"`: Specifies whether to train on CPU ("cpu") or GPU ("cuda").

### Training stability
Since v1.3 POME trains with settings that make repeated runs (different seeds) yield nearly identical embeddings: the encoding table is initialized from the spectral embedding of the bipartite sample-value graph, the sample and value rows are kept frozen, the variable encodings e_j are learned but anchored to their spectral initialization, the output geometry is anchored to its leading spectral components, and the learning rate decays to zero. Set `train_variable_encodings=False` for a fully frozen table, and `init="random", anchor_weight=0, gram_anchor_weight=0, lr_schedule="constant"` to reproduce POME ≤ 1.2.
- `init : str = "spectral"`: Initialization of the encoding table - `"spectral"` (deterministic, from a degree-normalized SVD of the sample-value graph) or `"random"` (POME ≤ 1.2).
- `anchor_weight : float = inf`: Penalty keeping the encoding table close to its initialization; `float("inf")` freezes the table, `0` lets it be learned freely.
- `train_variable_encodings : bool = True`: With a frozen table (`anchor_weight=inf`), keep the variable encodings e_j trainable while sample and value rows stay fixed. Because the value rows are fixed, the learned e_j are identifiable and can be compared directly (no shift/rescale ambiguity).
- `variable_anchor_weight : float = 0.1`: Penalty `w * mean_j ||e_j - e_j_init||^2` keeping the learned e_j near their spectral initialization; `0` learns them freely (not reproducible across seeds), `inf` freezes them.
- `gram_anchor_weight : float = 10.0`: Weight of the penalty pulling the pairwise cosine similarities of the sample embeddings towards those of the spectral reference; `0` disables it.
- `gram_reference_components : int = 8`: Number of leading spectral components forming that reference (`None` = all `embedding_dimension` components). Few components keep the dominant group structure and thus cluster separation.
- `lr_schedule : str = "cosine"`: Learning-rate schedule - `"cosine"` or `"linear"` decay to zero, or `"constant"` (POME ≤ 1.2).
- `negatives : str = "sampled"`: Negatives of the link-prediction loss - one sampled alternative value per positive pair and epoch (`"sampled"`), or all alternative values with equal weight (`"all"`, deterministic loss).

### Data handling
- `type_column : str = "type"`: Name of the column storing the variable types.
- `na_encoding : float = -99.0`: The float encoding value of missing data. Actual `NaN` entries are not supported and raise.
- `discretization_type : str = "z"`: How continuous variables are binned - `"z"` (z-score bins) or `"nonlinear"` (signed-power bins).
- `bins_per_continuous : int = 15`: Number of bins per continuous variable. With `"z"` discretization, only 3, 7, 11 and 15 are supported; other values raise.

### Imputation
- `enable_imputation : bool = False`: Set this to true if you want to use POME for imputation after training. It has to be set at construction time.
- `regressor_inputs : str = "invariant"`: Inputs of the regression head used for numeric imputation. `"invariant"` uses the sample node's normalized input feature and the numeric concept encoding of the variable (see the note on variable encodings below); `"raw"` uses the raw learned encodings (behaviour of POME ≤ 1.1).

### Inductive epoch tuning
With `inductive=True`, `fit()` does not train for a fixed number of epochs. It first runs a cross-validation that holds out whole samples and picks the epoch count that best generalizes to unseen ones, using a label-free train-vs-held-out effective-rank gap as the stopping signal. The selected value is stored in `_optimal_epochs`, while `epochs` serves as the upper cap and is restored after fitting.
- `inductive : bool = False`: Enables the CV-based epoch selection.

### Checkpointing & monitoring
All of these are disabled by default and add no cost to the training loop when unused.
- `file_name : str = None`, `output_path : str = None`: Filename prefix and output directory for saved artifacts.
- `epoch_checkpoints : int = -1`: Dump the whole fitted embedder via joblib every N epochs. Requires `file_name`.
- `embedding_csv_epochs : int = -1`: Write the current sample, variable and bin embeddings to CSV every N epochs.
- `epoch_callback = None`: Callable invoked as `epoch_callback(epoch, autoencoder)` after every epoch. It persists nothing itself - the caller decides what, if anything, to snapshot.


## Functions

After initializing the Embedder object, the main functions for using POME are:

- `fit(X, y=None)`: Training POME on the given input dataframe, with the input format as specified above.
- `get_embeddings(raw=False)`: Return computed embeddings in dataframe format. Output is a four-tuple of sample embeddings (position 0), variable embeddings (position 1), continuous-bin embeddings (position 2), and a mapping from sample name to its row in the attention matrix (position 3). Variable embeddings are the learned variable encodings e_j when the value encodings are frozen (`anchor_weight=inf`, the default), and otherwise the mean of each variable's normalized value-node inputs (same definition for numeric and categorical variables); bin embeddings are centred and scale-normalized. `raw=True` returns the raw stored encodings in every case (behaviour of POME ≤ 1.1) — see the note below before using them with a learnable table.
- `get_value_node_embeddings(level="input")`: Embeddings of all value nodes (categories and numeric bins), indexed `"variable=value"`. `level="input"` gives the normalized initial features fed into the encoder, `level="latent"` the encoder outputs (same space as the sample embeddings).
- `get_numeric_concept_embeddings()`: Learned shared-concept encodings of the numeric variables, centred and scale-normalized.
- `get_variable_centroid_norms()`: Norm of each variable's value-node centroid. Values close to zero mean that the variable's value nodes point in nearly opposite directions, so cosine similarities involving that variable are unstable.
- `transform(X)`: Embed new, unseen samples with the frozen trained encoder, without retraining. `X` must contain exactly the same variables as the training data plus the type column, and every sample must share at least one observed, known value with the training data — a sample whose values are all missing or unseen carries no signal and raises. Returns a `(num_new_samples, embedding_dimension)` dataframe.
- `impute_all(na_value : float)`: Imputes all missing values specified by `na_value` in the input dataset, and directly returns the imputed dataframe. Categorical values are imputed by scoring candidate values with the trained decoder, continuous ones by a regression head trained on the frozen embeddings. Requires `enable_imputation=True`.


### Note on variable encodings

The input feature of a value node is the normalized sum of a variable encoding and a value encoding (`e_j + e_c` for categories, `e_j + e_b` for numeric bins). The model only ever sees these normalized sums, so the individual learned encodings are not uniquely determined:

- For categorical variables, `e_j` can be shifted against its category encodings (and each sum rescaled) without changing the model. Raw categorical variable encodings therefore carry no meaning on their own.
- For numeric variables, bin encodings are shared across variables, which pins down the `e_j` up to one common shift and one global scale. `get_numeric_concept_embeddings()` removes both.
- All embeddings, including the sample embeddings, are only defined up to a global rotation. Cosine similarities and distances are unaffected, but raw coordinates of different runs should be aligned (e.g. by Procrustes) before being compared.

The default outputs of `get_embeddings()` and the other getters above are invariant to all of these ambiguities except the rotation.

With the default settings the value encodings `e_c` / `e_b` are frozen at their spectral initialization, which removes the shift and rescale ambiguities: the learned `e_j` are then identifiable and `get_embeddings()` returns them directly. The centroid representation above is used only when the table is learnable (`anchor_weight < inf`).


## License

POME is released under the GPL-3.0 license, see [LICENSE](./LICENSE).
