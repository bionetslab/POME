# POME: Learning partially observed mixed-type data embeddings 

POME is a graph-based representation-learning method for heterogeneous datasets that incorporates missingness structures into its low-dimensional embeddings. It is applicable to any tabular datasets consisting of both numeric- and categorical-type features.

## Installation
POME is implemented as a Python package and is easily installable from this repository by running
```
pip install -e .
```

## Input format
POME expects input data to be given in the form of a pandas dataframe object, with rows representing variables/features and columns representing samples. Missing data needs to be encoded by a unique numerical value. Furthermore, POME expects one column storing datatypes of the respective variables. An example dataset could have the following structure, with value -99 encoding missing data:
| | **Sample1** | **Sample2** | **Sample3** | **Type**
|----------|----------|----------|--------|----------|
| **VariableA**   | 0   | 1   | -99 | cat | 
| **VariableB**   | 3.14   | -0.1   | 2.5 | numerical |
| **VariableC**   | 0.3    | 1.2   | -99 | numerical |
| **VariableD**   | 1    | 0   | 2 | cat |

## Minimal working example
POME's core functionality is integrated into its `Embedder` class, which handles input transformation, training and output generation. In order to provide the user with an optimal, dataset-specific choice of parameters, we implemented an automated architecture search, whose output can then be passed to the `Embedder` class. A typical such workflow looks as follows:
```python
from pome import Embedder, run_architecture_search
import pandas as pd

if __name__ == "__main__":
    # Load data and set parameters.
    example_df = pd.read_csv("data/example.csv", index_col=0)
    NA_ENCODING = -99.0
    # Run final embedding with optimal architecture and parameters.
    embedder = Embedder(epochs=1000, non_informative_na=NA_ENCODING)
    embedder.fit(example_df)
    # Output stores low-dimensional embeddings for samples, variables, and bins of discretized numeric variables.
    sample_df, variable_df, bin_df, _ = embedder.get_embeddings()
    print("Computed sample embeddings: ", sample_df)
```
