import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import ToUndirected
import pandas as pd
import numpy as np
from torch_geometric.seed import seed_everything
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib
from gnn_embeddings.models import GraphAutoencoder
from gnn_embeddings.utils import compute_roc, repeat_pad_to_max_cols, bin_column_non_linear, bin_column_with_na_adjusted

def make_deterministic(seed=42):
    # 1. Basic seeding
    seed_everything(seed)
    
    # 2. PyTorch specific
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 3. Handle potential hashing randomness
    os.environ["PYTHONHASHSEED"] = str(seed)

class Embedder(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 embedding_dimension=32,
                 lr = 0.01,
                 bins_per_continuous=15,
                 epochs=500,
                 device="cpu",
                 layer_type="GAT",
                 type_column : str = "type",
                 non_informative_na : float = -99.0,
                 informative_nas : list = [],
                 file_name : str = None,
                 output_path : str = None,
                 discretization_type : str = "z",
                 enable_imputation : bool = False
                 ):
        self.embedding_dimension=embedding_dimension
        self.bins_per_continuous=bins_per_continuous
        self.epochs = epochs
        self.device = device
        self.lr = lr
        self.layer_type = layer_type
        self.num_nodes = None
        self.model = None
        self.type_column = type_column
        self.non_informative_na = non_informative_na
        self.informative_nas = informative_nas
        self.file_name = file_name
        self.output_path = output_path
        self.discretization_type = discretization_type
        self.enable_imputation = enable_imputation

    def fit(self, X, y=None):
        """Sklearn-based fit method for given tabular df.

        Args:
            X (tuple): Tuple storing the following data:
                - df (pd.DataFrame): Tabular df storing data matrix.
                - type_column (str): Name of column in df that stores variable type.
                - non_informative_na (float): Encoding value of non-informative NA.
                - informative_nas (list): List of encoding values of NAs that carry relevant information.
        """
        # Translate input data into graph based format.
        input_data = X.copy()
        discrete_data, graph_data, _, sample_node_dict, value_node_dict, var_value_dict, neg_edges_per_pair, variable_names, variable_embedding_ids, cont_bin_names, cont_bin_embedding_ids = self.data_to_graph(
                          input_data,
                          dtype_col=self.type_column,
                          ignore_na=self.non_informative_na,
                          keep_na=self.informative_nas
                          )

        # Save data structures potentially needed for imputation later on.
        self._neg_edges_per_pair = neg_edges_per_pair
        self._sample_node_dict = sample_node_dict
        self._variable_names = variable_names
        self._cont_bin_names = cont_bin_names
        self._X = input_data
        
        if self.enable_imputation:
            self._discretized_data = discrete_data
            self._var_value_dict = var_value_dict
            self._value_node_dict = value_node_dict
            self._cont_vars = set(self._X.index[self._X[self.type_column].isin(['numerical', 'cont'])])
            self._cat_vars = set(self._X.index[self._X[self.type_column].isin(['nominal', 'cat', 'ordinal'])])
        
        # Store actual dataframe without type column for potential imputation later.
        self._X.drop(columns=[self.type_column], inplace=True)

        # Compute node-based embeddings.
        node_embeddings, variable_embeddings, bin_embeddings, auc, ap = self.compute_node_embeddings(graph_data,
                                                                                     variable_embedding_ids,
                                                                                     cont_bin_embedding_ids)
        self._auc = auc
        self._ap = ap
        self._all_embeddings = node_embeddings
        self._variable_embeddings = variable_embeddings
        self._bin_embeddings = bin_embeddings
        
        return self
    
    def return_ap_score(self):
        if self._ap is None:
            raise ValueError("Error: model has not been trained yet, no AP exists.")
        else:
            return self._ap
    
    def decision_function(self, X):
        """Not used in this setup."""
        return np.array([])  # placeholder
    

    def get_sample_embeddings(self, format='pandas'):
        """Extract given set of precomputed embeddings.
        Args:
            sample_subset (list[int], optional): List of sample indices whose embeddings to return. Defaults to None.
        """
        if not hasattr(self, '_all_embeddings'):
            print("Run fit() first before extracting sample embeddings!")
        else:
            embedding_matrix = self._all_embeddings
            variable_matrix = self._variable_embeddings
            bin_matrix = self._bin_embeddings
            sample_rows = list(self._sample_node_dict.values())
            sample_embedding_matrix = embedding_matrix[sample_rows]
            sample_to_attention_index = {sample: row for row, sample in enumerate(self._sample_node_dict.keys())}
            if format=='pandas':
                # Initialize sample embedding matrix.
                sample_embedding_matrix = sample_embedding_matrix.numpy()
                embedding_cols = [f'dim_{i}' for i in range(self.embedding_dimension)]
                embedding_rows = list(self._X.columns)
                embedding_df = pd.DataFrame(sample_embedding_matrix, index=embedding_rows, columns=embedding_cols)
                # Init variable embeddings df.
                variable_rows = self._variable_names
                variable_cols = embedding_cols
                variable_df = pd.DataFrame(variable_matrix, index=variable_rows, columns=variable_cols)
                # Init cont-bin embeddings df.
                bin_rows = self._cont_bin_names
                bin_cols = embedding_cols
                bin_df = pd.DataFrame(bin_matrix, index=bin_rows, columns=bin_cols)
                return embedding_df, variable_df, bin_df, sample_to_attention_index
            elif format=='torch':
                return sample_embedding_matrix, variable_matrix, bin_matrix, sample_to_attention_index

    def impute_all(self, na_value : float, cont_imputation_mode='best_bin'):
        """Impute missing values for all samples in dataframe based on fitted node embeddings.

        Args:
            na_value (float): Impute all data entries that possess this missing value encoder.
        """
        samples = list(self._discretized_data.columns)
        aggregated_df = pd.DataFrame(index=self._X.index)
        for sample in samples:
            imputed_sample_df = self.impute_sample(sample, na_value, cont_imputation_mode)
            aggregated_df[sample] = imputed_sample_df[sample]
        return aggregated_df
    
    def impute_sample(self, sample_colum : str, na_value : float, cont_imputation_mode="best_bin"):
        """Impute missing values for given sample in dataframe based on fitted node embeddings.

        Args:
            sample_colum (str): Sample name whose values are supposed to be imputed.
            na_value (float): Impute all data entries that possess this missing value encoder.
        """
        if not self.enable_imputation:
            raise ValueError("Embedder object was created without imputation setup. Please reinstantiate Embedder object with enable_imputation=True.")
        
        if not sample_colum in self._discretized_data.columns:
            raise ValueError("Given sample column is not contained in input data.")
        
        imputed_df = self._X.copy()
        # Extract all positions (i.e. variables) in given sample that possess this missing value.
        impute_variables = self._discretized_data.index[self._discretized_data[sample_colum] == na_value].tolist()
        
        # Compute embedding similarities between sample and all to-impute variable nodes.
        for variable in impute_variables:
            # Retrieve for each sample & category node the respective embedding.
            potential_categories = [value for value, is_not_na in self._var_value_dict[variable] if is_not_na]
            category_indices = torch.tensor(
                [self._value_node_dict[value] for value in potential_categories],
                dtype=torch.long,
                device=self._all_embeddings.device
            )
            sample_index = torch.tensor(
                [self._sample_node_dict[sample_colum]] * len(potential_categories),
                dtype=torch.long,
                device=self._all_embeddings.device
            )

            # Edge index shape: [2, num_potential_categories]
            edge_index = torch.stack([sample_index, category_indices])

            # Get similarity scores from decoder
            with torch.no_grad():
                similarities = self._fitted_decoder(self._all_embeddings, edge_index)

            # Find category with highest similarity.
            best_idx = torch.argmax(similarities).item()
            closest_value = potential_categories[best_idx]
            # Extract value from category name.
            category_value = float(closest_value.split("=")[1])
            
            if variable in self._cat_vars:
                # Directly set imputed value at corresponding position in dataframe.
                imputed_df.loc[variable, sample_colum] = category_value

            elif variable in self._cont_vars:
                if cont_imputation_mode=='attention_based':
                    imputed_value = self._impute_cont_using_attention_average(sample_colum, variable)
                elif cont_imputation_mode=='weighted_bins':
                    imputed_value = self._impute_cont_using_weighted_average(sample_colum, variable)
                elif cont_imputation_mode=='best_bin':
                    bin_patients = self._discretized_data.columns[
                        self._discretized_data.loc[variable] == category_value
                        ].tolist()
                    if bin_patients:
                        imputed_value = self._X.loc[variable, bin_patients].mean()
                    else:
                        imputed_value = np.nan  # or global mean or other fallback
                else:
                    raise ValueError(f'Unknown continuous imputation mode: {cont_imputation_mode}')

                imputed_df.loc[variable, sample_colum] = imputed_value

        return imputed_df

    def impute_cont_using_best_bin_average(self, sample : str, variable : str, exclude_sample = False):
        # Retrieve for each sample & category node the respective embeddings.
        potential_categories = [value for value, is_not_na in self._var_value_dict[variable] if is_not_na]
        category_indices = torch.tensor(
            [self._value_node_dict[value] for value in potential_categories],
            dtype=torch.long,
            device=self._all_embeddings.device
        )
        sample_index = torch.tensor(
            [self._sample_node_dict[sample]] * len(potential_categories),
            dtype=torch.long,
            device=self._all_embeddings.device
        )

        # Torch tensor setup with edge index shape [2, num_potential_categories].
        edge_index = torch.stack([sample_index, category_indices])

        # Get sample-category similarity scores from fitted decoder.
        with torch.no_grad():
            similarities = self._fitted_decoder(self._all_embeddings, edge_index)

        # Find category with highest similarity.
        best_idx = torch.argmax(similarities).item()
        closest_value = potential_categories[best_idx]
        # Extract value from category name.
        category_value = float(closest_value.split("=")[1])

        best_bin_samples = set(self._discretized_data.columns[
                                   self._discretized_data.loc[variable] == category_value
                                   ])
        if exclude_sample:
            best_bin_samples = best_bin_samples - set(sample)

        if len(best_bin_samples) > 0:
            return self._X.loc[variable, list(best_bin_samples)].mean()
        else:
            return np.nan

    def impute_cont_using_weighted_average(self, sample : str, var : str, exclude_sample = False):
        sample_id = self._sample_node_dict[sample]
        # Only select non-NA bins for current variable.
        non_na_bins = [x[0] for x in self._var_value_dict[var] if x[1] == True]

        # Iterate once over all non-NA bins to compute softmax-ed similarities.
        sample_bin_edges = [[], []]
        for variable_bin in non_na_bins:
            bin_id = self._value_node_dict[variable_bin]
            sample_bin_edges[0].append(sample_id)
            sample_bin_edges[1].append(bin_id)
        # Run fitted decoder to obtain sample-bin similarities.
        sample_bin_sim = self._fitted_decoder(self._all_embeddings, sample_bin_edges)
        sample_bin_probs = F.softmax(sample_bin_sim, dim=0)

        # Again iterate over all non-NA bin and compute weighted imputation value, including attention weights.
        only_empty_bins = True
        imputed_value = 0.0
        for bin_id, variable_bin in enumerate(non_na_bins):
            sample_bin_prob = sample_bin_probs[bin_id]
            # Compute unweighted mean over all sample values in current bin.
            bin_value = float(variable_bin.split("=")[1])
            bin_patients = set(self._discretized_data.columns[
                self._discretized_data.loc[var] == bin_value
                ])
            if exclude_sample:
                bin_patients = bin_patients - {sample}
            if len(bin_patients)>0:
                bin_average = self._X.loc[var, list(bin_patients)].mean()
                only_empty_bins = False
                imputed_value += sample_bin_prob * bin_average
            else: # In case there are no samples in corresponding bin, ignore this bin.
                continue

        if only_empty_bins:
            return np.nan
        else:
            return float(imputed_value)

    def _move_self_tensors(self, device):
        """Move all tensor-like attributes of self to given device."""
        for name, value in self.__dict__.items():
            if hasattr(value, "to") and callable(value.to):
                try:
                    self.__dict__[name] = value.to(device)
                except Exception:
                    pass  # skip objects that cannot be moved

    def train_gnn(self, autoencoder, optimizer, graph_data, save_epochs=-1):
        neg_edges_list = [val for _, val in self._neg_edges_per_pair.items() if val.numel()>0]
        neg_edges_tensor = repeat_pad_to_max_cols(neg_edges_list) 
        
        neg_edges_tensor = neg_edges_tensor.to(self.device)
        graph_data.edge_index = graph_data.edge_index.to(self.device)
        graph_data.unique_edges = graph_data.unique_edges.to(self.device)

        B, _, max_cols = neg_edges_tensor.shape
    
        # 2. Pre-generate all random indices for all epochs at once
        # Resulting shape: (epochs, B)
        all_rand_indices = torch.randint(0, max_cols, (self.epochs, B), device=self.device)

        # Pre-allocate labels to avoid re-creating them in the loop
        pos_labels = torch.ones(graph_data.unique_edges.size(1), device=self.device)
        neg_labels = torch.zeros(B, device=self.device)
        combined_labels = torch.cat([pos_labels, neg_labels], dim=0)

        # Inside your training loop:
        for epoch in range(self.epochs):

            # --- PART 1: SAMPLING ---
            current_indices = all_rand_indices[epoch]
            neg_edge_index = neg_edges_tensor[torch.arange(B, device=self.device), :, current_indices].T
            pos_neg_combined = torch.cat([graph_data.unique_edges, neg_edge_index], dim=1)

            # --- PART 2: FORWARD PASS ---
            pos_neg_similarities = autoencoder(graph_data.edge_index, pos_neg_combined)
            loss = F.binary_cross_entropy(pos_neg_similarities, combined_labels)

            # --- PART 3: BACKWARD & STEP ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f"Step {epoch}: loss = {loss.item()}")
                #print(f"Epoch {epoch} | Sample: {sample_time:.2f}ms | Forward: {forward_time:.2f}ms | Backward: {backward_time:.2f}ms")
                
            if self.file_name and save_epochs > 0 and (epoch+1) % save_epochs == 0 and epoch > 0:
                # Save currently fitted decoder and embeddings to self.
                node_embeddings, _, _ = autoencoder.get_embeddings(graph_data.edge_index)
                self._all_embeddings = node_embeddings
                # Save trained decoder for later potential imputation.
                fitted_decoder = autoencoder.get_decoder()
                self._fitted_decoder = fitted_decoder
                
                # Move everything to CPU
                self._move_self_tensors("cpu")
                graph_data = graph_data.to("cpu")

                # Save checkpoint
                save_path = os.path.join(self.output_path, f"{self.file_name}_epoch_{epoch}.joblib")
                neg_edges_list_auc = [val for _, val in self._neg_edges_per_pair.items()]
                auc, ap = compute_roc(graph_data, neg_edges_list_auc, self._all_embeddings, self._fitted_decoder)
                self._auc = auc
                self._ap = ap
                joblib.dump(self, save_path)
                print(f"Saved checkpoint to {save_path}.")

                # Move back to GPU
                self._move_self_tensors(self.device)
                graph_data = graph_data.to(self.device)

    
    def compute_node_embeddings(self, graph_data, variable_embeddings_ids, bin_embedding_ids):
        """Core function for fitting optimal node embeddings.
        """
        # Extract number of required embedding vectors.
        num_embeddings = int(torch.max(graph_data.node_to_embeddings))+1
        
        # Shift graph representation data to GPU if possible.
        graph_data = graph_data.to(self.device)
        variable_embeddings_ids = variable_embeddings_ids.to(self.device)
        bin_embedding_ids = bin_embedding_ids.to(self.device)

        # Init GraphAutoencoder model.
        self.model = GraphAutoencoder(num_embeddings=num_embeddings, 
                                      embedding_dim=self.embedding_dimension, 
                                      encoder_layer=self.layer_type,
                                      node_to_embeddings_index=graph_data.node_to_embeddings,
                                      variable_embeddings_ids=variable_embeddings_ids,
                                      bin_embedding_ids=bin_embedding_ids).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Train model using link prediction loss.
        self.train_gnn(self.model, optimizer, graph_data)

        # Extract node embeddings from fitted model.
        self.model.eval()
        node_embeddings, variable_embeddings, bin_embeddings = self.model.get_embeddings(graph_data.edge_index)

        # Save trained decoder for later potential imputation.
        fitted_decoder = self.model.get_decoder().to('cpu')
        self._fitted_decoder = fitted_decoder

        # Run post-training validation, to check embedding quality.
        graph_data = graph_data.to('cpu')
        node_embeddings = node_embeddings.to('cpu')
        variable_embeddings = variable_embeddings.to('cpu')
        bin_embeddings = bin_embeddings.to('cpu')
        neg_edges_list = [val for _, val in self._neg_edges_per_pair.items()]
        auc, ap = compute_roc(graph_data, neg_edges_list, node_embeddings, fitted_decoder)

        return node_embeddings, variable_embeddings, bin_embeddings, auc, ap

    def data_to_graph(self, all_data, dtype_col: str, ignore_na, keep_na):

        # Extract sample and variable information from tabular df.
        data = all_data.copy()
        num_variables = len(data.index)
        samples = list(data.columns)
        samples.remove(dtype_col)
        variables = list(data.index)

        # Identify and bin numerical variables.
        continuous_vars = [var for var, row in data.iterrows() if (row[dtype_col] == 'numerical' or row[dtype_col] == "cont")]
        variable_types = data[dtype_col].to_list()
        data.drop(columns=[dtype_col], inplace=True)

        if self.discretization_type == "z":
            data.loc[continuous_vars] = data.loc[continuous_vars].apply(
                lambda row: bin_column_with_na_adjusted(row, self.bins_per_continuous, ignore_na, keep_na), axis=1)
        elif self.discretization_type == "nonlinear":
            data.loc[continuous_vars] = data.loc[continuous_vars].apply(
                lambda row: bin_column_non_linear(row, self.bins_per_continuous, ignore_na, keep_na), axis=1)
        else:
            raise ValueError(f"Unknown discretization mode: {self.discretization_type}")
        
        data[dtype_col] = variable_types
        
        # Create representing data structures for graph representation of tabular data.
        num_samples = len(samples)
        value_node_ids = {}
        sample_node_ids = {sample: idx for idx, sample in enumerate(samples)}
        edges = [[], []]
        num_nodes = num_samples
        var_value_dict = {var: set() for var in variables}

        # Record sample-variable pairs that actually possess positive edge, i.e. non-NA value.
        positive_edge_pairs = []        

        # Initialize node-to-embeddings index, storing which variable- and value-level embeddings
        # from the global embeddings matrix are supposed to be used.
        node_to_embeddings = {idx : [idx, idx] for _, idx in sample_node_ids.items()}
        
        variable_to_embeddings = {variable : idx+num_samples for idx, variable in enumerate(variables)}
        cont_bin_to_embeddings = {bin : bin+num_samples+num_variables for bin in range(self.bins_per_continuous)}
        na_bin_to_embeddings = {na : idx+num_samples+num_variables+self.bins_per_continuous for idx,na in enumerate(keep_na)}
        cat_bin_counter = self.bins_per_continuous + num_samples + num_variables + len(keep_na)
        cat_bins_to_embeddings = dict()

        for var in data.index:
            for sample in samples:
                value = data.loc[var, sample]
                if value == ignore_na:
                    continue

                value_label = f"{var}={value}"
                is_value_na = (value in keep_na)

                var_value_dict[var].add((value_label, not is_value_na))

                # Record positive sample-variable pair.
                positive_edge_pairs.append((var, sample, value_label))

                if value_label not in value_node_ids:
                    value_node_ids[value_label] = num_nodes
                    num_nodes += 1

                value_node_id = value_node_ids[value_label]
                sample_node_id = sample_node_ids[sample]

                edges[0].append(sample_node_id)
                edges[1].append(value_node_id)
                
                # Save node-to-embedding relationship for value nodes.
                if value in keep_na:
                    node_to_embeddings[value_node_id] = [variable_to_embeddings[var], na_bin_to_embeddings[value]]
                elif var in continuous_vars:
                    node_to_embeddings[value_node_id] = [variable_to_embeddings[var], cont_bin_to_embeddings[value]]
                else:
                    if value_label in cat_bins_to_embeddings:
                        node_to_embeddings[value_node_id] = [variable_to_embeddings[var], cat_bins_to_embeddings[value_label]]
                    else:
                        cat_bins_to_embeddings[value_label] = cat_bin_counter
                        cat_bin_counter += 1
                        node_to_embeddings[value_node_id] = [variable_to_embeddings[var], cat_bins_to_embeddings[value_label]]
                    
        # Prepare node-to-embedding mapping for summing up corresponding embeddings per node.
        sorted_values = [
            node_to_embeddings[key] 
        for key in sorted(node_to_embeddings.keys())
        ]
        node_to_embedding_tensor = torch.tensor(sorted_values, dtype=torch.long)
        variable_embedding_ids = torch.tensor(list(variable_to_embeddings.values()), dtype=torch.long)
        cont_bin_embedding_ids = torch.tensor(list(cont_bin_to_embeddings.values()), dtype=torch.int)
        variable_names = list(variable_to_embeddings.keys())
        cont_bin_names = [f"bin_{value}" for value in cont_bin_to_embeddings.keys()]
        
        # Transform graph representation into PyTorch format.
        edge_index = torch.tensor(edges, dtype=torch.long)
        graph_data = Data(edge_index=edge_index)
        graph_data.unique_edges = edge_index
        graph_data = ToUndirected()(graph_data)
        graph_data.num_samples = num_samples
        graph_data.num_nodes = num_nodes
        graph_data.node_to_embeddings = node_to_embedding_tensor

        # Precompute all potential negative edges to later draw from. For each sample-variable pair that we have a
        # positive ege, we also record all possible negative edges (i.e. not for NA values).
        neg_edges_per_pair = dict()
        for var, sample, pos_value in positive_edge_pairs:
            negative_edges = [[], []]
            sample_id = sample_node_ids[sample]
            for value, is_not_value_na in var_value_dict[var]:
                value_id = value_node_ids[value]
                if value != pos_value and is_not_value_na:
                    negative_edges[0].append(sample_id)
                    negative_edges[1].append(value_id)
            neg_edges_per_pair[(sample, var, pos_value)] = (torch.tensor(negative_edges, dtype=torch.long, device=self.device))

        data.drop(columns=[dtype_col], inplace=True)
        return (data, graph_data, num_variables, sample_node_ids, value_node_ids, 
                var_value_dict, neg_edges_per_pair, variable_names, variable_embedding_ids,
                cont_bin_names, cont_bin_embedding_ids)

if __name__ == "__main__":
    
    NON_INFORMATIVE_NA = -99.0
    INFORMATIVE_NAS = []
    device = "cuda"
    FILE_NAME = "data/input/mimic_aggregated_with_targets.tsv"
    OUT_PATH = "/data/bionets/xa39zypy/graph_based_embeddings.git/data/preprocessed/graph_based/MIMIC/with_target_embeddings"
    dataset = 'MIMIC'
    df = pd.read_csv(FILE_NAME, index_col=0, sep='\t')
    
    input_params = {'non_informative_na' : NON_INFORMATIVE_NA, 'informative_nas' : INFORMATIVE_NAS, "device" : device}
    opt = dict()
    embedder_params = opt | input_params
    
    embedder_params["embedding_dimension"]=16
    for i in range(10):
        print("Running embedding = ", i)
        make_deterministic(i)
        embedder = Embedder( 
                           epochs=2000,
                           **embedder_params
                           )
        embedder.fit(df.copy())
        
        sample_df, var_df, bin_df, _ = embedder.get_sample_embeddings()
        emb_dim = embedder_params["embedding_dimension"]

        sample_df.to_csv(
            os.path.join(OUT_PATH, f"{dataset}_samples_{emb_dim}_{i}.tsv"),
            sep="\t",
            index=True
        )
        var_df.to_csv(
            os.path.join(OUT_PATH, f"{dataset}_variables_{emb_dim}_{i}.tsv"),
            sep="\t",
            index=True
        )
        bin_df.to_csv(
            os.path.join(OUT_PATH, f"{dataset}_bins_{emb_dim}_{i}.tsv"),
            sep="\t",
            index=True
        )
    
    embedder_params["embedding_dimension"]=32
    for i in range(10):
        print("Running embedding = ", i)
        make_deterministic(i)
        embedder = Embedder( 
                           epochs=2000,
                           **embedder_params
                           )
        embedder.fit(df.copy())
        
        sample_df, var_df, bin_df, _ = embedder.get_sample_embeddings()
        emb_dim = embedder_params["embedding_dimension"]

        sample_df.to_csv(
            os.path.join(OUT_PATH, f"{dataset}_samples_{emb_dim}_{i}.tsv"),
            sep="\t",
            index=True
        )
        var_df.to_csv(
            os.path.join(OUT_PATH, f"{dataset}_variables_{emb_dim}_{i}.tsv"),
            sep="\t",
            index=True
        )
        bin_df.to_csv(
            os.path.join(OUT_PATH, f"{dataset}_bins_{emb_dim}_{i}.tsv"),
            sep="\t",
            index=True
        )
    
    embedder_params["embedding_dimension"]=64
    for i in range(10):
        print("Running embedding = ", i)
        make_deterministic(i)
        embedder = Embedder( 
                           epochs=2000,
                           **embedder_params
                           )
        embedder.fit(df.copy())
        
        sample_df, var_df, bin_df, _ = embedder.get_sample_embeddings()
        emb_dim = embedder_params["embedding_dimension"]

        sample_df.to_csv(
            os.path.join(OUT_PATH, f"{dataset}_samples_{emb_dim}_{i}.tsv"),
            sep="\t",
            index=True
        )
        var_df.to_csv(
            os.path.join(OUT_PATH, f"{dataset}_variables_{emb_dim}_{i}.tsv"),
            sep="\t",
            index=True
        )
        bin_df.to_csv(
            os.path.join(OUT_PATH, f"{dataset}_bins_{emb_dim}_{i}.tsv"),
            sep="\t",
            index=True
        )
