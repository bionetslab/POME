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
from pome.models import GraphAutoencoder
from pome.utils import compute_roc, repeat_pad_to_max_cols, bin_column_non_linear, bin_column_with_na_adjusted, signed_power_bins, get_zscore_bins

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
                 na_encoding : float = -99.0,
                 informative_nas : list = [],
                 file_name : str = None,
                 output_path : str = None,
                 discretization_type : str = "z",
                 enable_imputation : bool = False,
                 epoch_checkpoints : int = -1,
                 ):
        self.embedding_dimension=embedding_dimension
        self.bins_per_continuous=bins_per_continuous
        self.epochs = epochs
        self.device = device
        self.lr = lr
        self.layer_type = layer_type
        self.model = None
        self.type_column = type_column
        self.non_informative_na = na_encoding
        self.informative_nas = informative_nas
        self.file_name = file_name
        self.output_path = output_path
        self.discretization_type = discretization_type
        self.enable_imputation = enable_imputation
        self.save_epochs = epoch_checkpoints

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
        discrete_data, graph_data, _, sample_node_dict, value_node_dict, var_value_dict, neg_edges_per_pair, variable_names, variable_embedding_ids, cont_bin_names, cont_bin_embedding_ids, bin_stats = self.data_to_graph(
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
        self._value_node_dict = value_node_dict
        self._bin_stats = bin_stats

        if self.enable_imputation:
            self._discretized_data = discrete_data
            self._var_value_dict = var_value_dict
            self._cat_vars = set(self._X.index[self._X[self.type_column].isin(['nominal', 'cat', 'ordinal'])])

        # Store actual dataframe without type column for potential imputation later.
        self._cont_vars = set(self._X.index[self._X[self.type_column].isin(['numerical', 'cont'])])
        self._X.drop(columns=[self.type_column], inplace=True)

        # Compute node-based embeddings.
        node_embeddings, variable_embeddings, bin_embeddings, auc, ap = self.compute_node_embeddings(graph_data,
                                                                                     variable_embedding_ids,
                                                                                     cont_bin_embedding_ids)
        self._graph_data = graph_data  # graph_data is on CPU after compute_node_embeddings
        self._auc = auc
        self._ap = ap
        self._all_embeddings = node_embeddings
        self._variable_embeddings = variable_embeddings
        self._bin_embeddings = bin_embeddings
        
        return self
    
    def return_ap_score(self):
        return self._ap
    
    def decision_function(self, X):
        """Not used in this setup."""
        return np.array([])  # placeholder
    

    def get_embeddings(self):
        """Extract given set of precomputed embeddings.
        Args:
            sample_subset (list[int], optional): List of sample indices whose embeddings to return. Defaults to None.
        """
        if not hasattr(self, '_all_embeddings'):
            raise ValueError("Run fit() first before extracting sample embeddings!")
        else:
            embedding_matrix = self._all_embeddings
            variable_matrix = self._variable_embeddings
            bin_matrix = self._bin_embeddings
            sample_rows = list(self._sample_node_dict.values())
            sample_embedding_matrix = embedding_matrix[sample_rows]
            sample_to_attention_index = {sample: row for row, sample in enumerate(self._sample_node_dict.keys())}
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

    def impute_all(self, na_value : float):
        """Impute missing values for all samples in dataframe based on fitted node embeddings.

        Args:
            na_value (float): Impute all data entries that possess this missing value encoder.
        """
        
        if self.enable_imputation == False:
            raise ValueError("Embedder object has not been initialized with enable_imputation=True. Please re-train.")
        
        samples = list(self._discretized_data.columns)
        aggregated_df = pd.DataFrame(index=self._X.index)
        for sample in samples:
            imputed_sample_df = self.impute_sample(sample, na_value)
            aggregated_df[sample] = imputed_sample_df[sample]
        return aggregated_df
    
    def impute_sample(self, sample_colum : str, na_value : float):
        """Impute missing values for given sample in dataframe based on fitted node embeddings.

        Args:
            sample_colum (str): Sample name whose values are supposed to be imputed.
            na_value (float): Impute all data entries that possess this missing value encoder.
        """
        if not self.enable_imputation:
            raise ValueError("Embedder object was created without imputation setup. Please reinstantiate Embedder object with enable_imputation=True.")
        
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
                bin_patients = self._discretized_data.columns[
                    self._discretized_data.loc[variable] == category_value
                    ].tolist()
                if bin_patients:
                    imputed_value = self._X.loc[variable, bin_patients].mean()
                else:
                    imputed_value = np.nan  # or global mean or other fallback

                imputed_df.loc[variable, sample_colum] = imputed_value

        return imputed_df


    def _remap_to_populated_bin(self, var, bin_id):
        """Return the nearest bin index to bin_id that has a node in the training graph."""
        for delta in range(1, self.bins_per_continuous):
            for candidate in (bin_id - delta, bin_id + delta):
                if 0 <= candidate < self.bins_per_continuous:
                    if f"{var}={float(candidate)}" in self._value_node_dict:
                        return candidate
        return None

    def _get_bin_id(self, variable, value):
        """Map a single continuous value to its training-time bin ID."""
        stats = self._bin_stats[variable]
        if stats['type'] == 'z':
            std = stats['std']
            z = 0.0 if std == 0 else (value - stats['mean']) / std
            result = pd.cut([z], bins=get_zscore_bins(self.bins_per_continuous), labels=False)
            bin_val = result[0]
            return None if pd.isna(bin_val) else int(bin_val)
        elif stats['type'] == 'nonlinear':
            return int(np.digitize(value, stats['edges'][1:-1], right=True))
        return None

    def _sample_value_node_pairs(self, input_data, sample):
        """Map one new sample to the training value nodes it connects to.

        Reproduces the edge-construction logic used during fit(): continuous values
        are mapped to their training-time bin (with empty-bin remapping), NA sentinels
        are skipped, and unseen categories are dropped.

        Returns:
            list[tuple[str, int]]: (variable, value_node_index) pairs for each observed,
                                   in-vocabulary value of the sample.
        """
        pairs = []
        for var in input_data.index:
            raw = input_data.loc[var, sample]
            if pd.isna(raw):
                raise ValueError("Actual NA entry detected in input dataframe.")
            value = float(raw)
            if value == self.non_informative_na:
                continue

            if var in self._cont_vars and value not in self.informative_nas:
                bin_id = self._get_bin_id(var, value)
                if bin_id is None:
                    continue
                value_label = f"{var}={float(bin_id)}"
                if value_label not in self._value_node_dict:
                    remapped = self._remap_to_populated_bin(var, bin_id)
                    if remapped is None:
                        continue
                    value_label = f"{var}={float(remapped)}"
            else:
                value_label = f"{var}={value}"
                if value_label not in self._value_node_dict:
                    continue

            pairs.append((var, self._value_node_dict[value_label]))
        return pairs

    def _augmented_edge_index(self, pairs_per_sample, base_idx, device):
        """Build the augmented edge_index that injects new sample nodes into the training graph.

        Only ``value_node -> new_node`` edges are added: each new sample *receives* messages
        from the frozen value nodes but never sends any back. This keeps the shared value-node
        representations identical to training and lets every new sample be encoded in a single
        pass without perturbing the training nodes or interfering with each other.

        Args:
            pairs_per_sample (list[list[tuple[str, int]]]): per-new-sample (var, value_node) pairs.
            base_idx (int): node index assigned to the first new sample (subsequent samples
                            are base_idx+1, base_idx+2, ...).
            device: torch device.

        Returns:
            torch.Tensor | None: augmented edge_index, or None if no new edges could be built.
        """
        value_src, new_dst = [], []
        for offset, pairs in enumerate(pairs_per_sample):
            node = base_idx + offset
            for _, value_node in pairs:
                # PyG convention: a message flows src -> dst, so value_node -> new_node lets
                # the new sample aggregate from its value nodes (and never the reverse).
                value_src.append(value_node)
                new_dst.append(node)
        if not value_src:
            return None
        recv_edges = torch.tensor([value_src, new_dst], dtype=torch.long, device=device)
        training_edge_index = self._graph_data.edge_index.to(device)
        return torch.cat([training_edge_index, recv_edges], dim=1)

    def _frozen_encode(self, augmented_embeds, augmented_edge_index):
        """Normalize node inputs, run the frozen encoder, and L2-normalize the latents.

        Mirrors GraphAutoencoder.get_embeddings exactly so training (transductive) and
        inductive nodes are processed identically.
        """
        norms = augmented_embeds.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
        augmented_embeds = augmented_embeds / norms
        latent = self.model.encoder(augmented_embeds, augmented_edge_index)
        norms = latent.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
        return latent / norms

    def transform(self, X):
        """Generate embeddings for new, unseen samples using the frozen trained encoder.

        The trained model weights are never updated (no retraining). Each new sample is added
        to the training graph as a node connected to the training value nodes it observes, with
        edges pointing *only* from value nodes to the new sample. Because the new samples never
        send messages back, the shared value-node representations stay identical to training
        and all new samples are encoded in a single forward pass without affecting the training
        nodes or one another. The new node's own ("self-loop") input embedding is set to zeros:
        the trained encoder assigns negligible weight to a sample node's self-loop, so the
        embedding is determined by which value nodes the sample connects to (its data), and a
        data-derived initialization was found to have no measurable effect.

        Empirically this places inductive embeddings in the same distribution as the
        transductive training embeddings (Rosenbaum cross-match z ~ 0). Adding the new samples
        with bidirectional edges instead perturbs the value nodes and is markedly
        out-of-distribution.

        Every sample must share at least one observed, in-vocabulary value with the training
        data; an edgeless sample (all values missing / unseen) has no signal and raises.

        Args:
            X (pd.DataFrame): New samples in the same format as the training data
                              (rows=variables, columns=new_samples+type_column).

        Returns:
            pd.DataFrame: Embedding matrix with shape (num_new_samples, embedding_dimension),
                          indexed by sample name.
        """
        if not hasattr(self, '_graph_data'):
            raise ValueError("Run fit() first before calling transform()!")

        input_data = X.copy()
        new_samples = [c for c in input_data.columns if c != self.type_column]
        input_data.drop(columns=[self.type_column], inplace=True)

        new_vars = set(input_data.index)
        train_vars = set(self._variable_names)
        if new_vars != train_vars:
            missing = sorted(train_vars - new_vars)
            extra = sorted(new_vars - train_vars)
            parts = []
            if missing:
                parts.append(f"missing from input: {missing}")
            if extra:
                parts.append(f"not seen during training: {extra}")
            raise ValueError("Variable mismatch between transform input and training data — " + "; ".join(parts))

        # Resolve which training value nodes each new sample connects to. Every sample must
        # have at least one valid edge, otherwise it carries no signal for the encoder.
        pairs_per_sample = [self._sample_value_node_pairs(input_data, s) for s in new_samples]
        edgeless = [s for s, pairs in zip(new_samples, pairs_per_sample) if not pairs]
        if edgeless:
            raise ValueError(
                f"No valid edges for sample(s) {edgeless}: none of their values are observed, "
                "in-vocabulary values shared with the training data, so they have no signal. "
                "Every sample must share at least one value with the training data.")

        num_existing_nodes = self._graph_data.num_nodes
        device = next(self.model.parameters()).device
        self.model.eval()

        with torch.no_grad():
            # Frozen training-node input embeddings (component_1 + component_2). New sample
            # nodes get a zero self-loop feature (the encoder ignores it; the embedding comes
            # entirely from the value nodes the sample connects to).
            c1 = self.model.node_embeddings(self.model.node_to_embeddings[:, 0].to(device))
            c2 = self.model.node_embeddings(self.model.node_to_embeddings[:, 1].to(device))
            training_embeds = c1 + c2

            new_sample_embeds = torch.zeros(
                len(new_samples), self.embedding_dimension,
                device=device, dtype=training_embeds.dtype)
            augmented_embeds = torch.cat([training_embeds, new_sample_embeds], dim=0)
            augmented_edge_index = self._augmented_edge_index(
                pairs_per_sample, num_existing_nodes, device)
            latent = self._frozen_encode(augmented_embeds, augmented_edge_index)
            new_latent = latent[num_existing_nodes:].cpu()

        embedding_cols = [f'dim_{i}' for i in range(self.embedding_dimension)]
        return pd.DataFrame(new_latent.numpy(), index=new_samples, columns=embedding_cols)

    def _move_self_tensors(self, device):
        """Move all tensor-like attributes of self to given device."""
        for name, value in self.__dict__.items():
            if hasattr(value, "to") and callable(value.to):
                self.__dict__[name] = value.to(device)

    def train_gnn(self, autoencoder, optimizer, graph_data):
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
                
            if self.file_name and self.save_epochs > 0 and (epoch+1) % self.save_epochs == 0 and epoch > 0:
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

        # Compute per-variable bin stats for later inductive inference on new samples.
        bin_stats = {}
        for var in continuous_vars:
            col = data.loc[var].copy().astype(float)
            valid = ~((col == ignore_na) | col.isin(keep_na) | col.isna())
            non_na = col[valid]
            if self.discretization_type == "z":
                if len(set(non_na)) == 1:
                    bin_stats[var] = {'type': 'z', 'mean': float(non_na.iloc[0]), 'std': 1.0}
                else:
                    bin_stats[var] = {'type': 'z', 'mean': float(non_na.mean()), 'std': float(non_na.std(ddof=0))}
            elif self.discretization_type == "nonlinear":
                if len(set(non_na)) == 1:
                    val = float(non_na.iloc[0])
                    bin_stats[var] = {'type': 'nonlinear', 'edges': np.array([-np.inf, val, np.inf])}
                else:
                    edges, _ = signed_power_bins(non_na.values, n_bins=self.bins_per_continuous)
                    bin_stats[var] = {'type': 'nonlinear', 'edges': edges}

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
                cont_bin_names, cont_bin_embedding_ids, bin_stats)
