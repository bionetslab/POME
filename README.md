# POME: Learning partially observed mixed-type data embeddings 
![Tests](https://github.com/bionetslab/POME/actions/workflows/tests.yaml/badge.svg)
![Coverage](https://raw.githubusercontent.com/bionetslab/POME/main/badges/coverage.svg)

POME is a graph-based representation-learning method for heterogeneous datasets that incorporates missingness structures into the computation of low-dimensional sample and variable embeddings. It is applicable to any tabular datasets consisting of both numeric- and categorical-type features, where missing data patterns are supposed to be taken into account.

## Installation
POME is implemented as a Python package and is easily installable from this repository by running
```
pip install -e .
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

## Parameters

POME's Embedder class allows for the specification of the following parameters:

- `embedding_dimension : int = 32`: Specifies the number of dimensions of the sample & variable embeddings learned by POME.
- `epochs : int = 500`: Sets the number of epochs that POME is supposed to be trained.
- `device : str = "cpu"`: Specifies whether to train on CPU ("cpu") or GPU ("cuda").
- `na_encoding : float = -99.0`: The float encoding value of missing data.
- `enable_imputation : bool = False`: Set this to true if you want to use POME for imputation after training.


## Functions

After initializing the Embedder object, the main three functions for using POME are:

- `fit(self, X, y=None)`: Training POME on the given input dataframe, with the input format as specified above.
- `get_embeddings(self, format='pandas')`: Return computed embeddings of samples and variables in dataframe format. Output is a four-tuple with sample embeddings in at position 0, and variable embeddings at position 1.
- `impute_all(self, na_value : float)`: Imputes all missing values specified by `na_value` in the input dataset, and directly returns the imputed dataframe.



