# POME: Learning partially observed mixed-type data embeddings 
![Tests](https://github.com/bionetslab/POME/actions/workflows/tests.yaml/badge.svg)
![Coverage Status](./coverage.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyPI](https://img.shields.io/pypi/v/pome-py?color=orange)

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

### Data handling
- `type_column : str = "type"`: Name of the column storing the variable types.
- `na_encoding : float = -99.0`: The float encoding value of missing data. Actual `NaN` entries are not supported and raise.
- `discretization_type : str = "z"`: How continuous variables are binned - `"z"` (z-score bins) or `"nonlinear"` (signed-power bins).
- `bins_per_continuous : int = 15`: Number of bins per continuous variable. With `"z"` discretization, only 3, 7, 11 and 15 are supported; other values raise.

### Imputation
- `enable_imputation : bool = False`: Set this to true if you want to use POME for imputation after training. It has to be set at construction time.

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
- `get_embeddings()`: Return computed embeddings in dataframe format. Output is a four-tuple of sample embeddings (position 0), variable embeddings (position 1), continuous-bin embeddings (position 2), and a mapping from sample name to its row in the attention matrix (position 3).
- `transform(X)`: Embed new, unseen samples with the frozen trained encoder, without retraining. `X` must contain exactly the same variables as the training data plus the type column, and every sample must share at least one observed, known value with the training data — a sample whose values are all missing or unseen carries no signal and raises. Returns a `(num_new_samples, embedding_dimension)` dataframe.
- `impute_all(na_value : float)`: Imputes all missing values specified by `na_value` in the input dataset, and directly returns the imputed dataframe. Categorical values are imputed by scoring candidate values with the trained decoder, continuous ones by a regression head trained on the frozen embeddings. Requires `enable_imputation=True`.


## License

POME is released under the GPL-3.0 license, see [LICENSE](./LICENSE).
