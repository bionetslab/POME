from torch_geometric.nn import GATConv, GCNConv, SAGEConv
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, layer_type):
        super(GraphEncoder, self).__init__()
        
        layer_type = layer_type.upper()

        # Pick the right layer constructor
        if layer_type == "GAT":
            Layer = GATConv
        else:
            raise ValueError(f"Unknown layer_type: {layer_type}. Choose from ['GCN', 'GAT', 'SAGE']")

        self.convs = torch.nn.ModuleList()

        # First hidden layer
        self.convs.append(Layer(input_dim, hidden_dims[0]))

        # Additional hidden layers
        for i in range(1, len(hidden_dims)):
            self.convs.append(Layer(hidden_dims[i - 1], hidden_dims[i]))

        # Output layer
        self.convs.append(Layer(hidden_dims[-1], output_dim))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            #x = F.dropout(x, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

class MLPDecoder(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Concatenate two given node embeddings.
        self.layer1 = nn.Linear(input_dim*2, input_dim)
        # Model predicts single similarity value which is scaled to unit interval.
        self.layer2 = nn.Linear(input_dim, 1)
        
    def forward(self, node_embeddings, edge_index):
        # Extract contained node pairs.
        u, v = edge_index
        # Concat node embeddings for all edges.
        edge_features = torch.cat([node_embeddings[u], node_embeddings[v]], dim=1)
        x = F.relu(self.layer1(edge_features))
        x = self.layer2(x)
        return torch.sigmoid(x).squeeze()

# Graph Autoencoder
class GraphAutoencoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, encoder_layer, node_to_embeddings_index,
                 variable_embeddings_ids, bin_embedding_ids):
        super().__init__()

        # Learnable initial normalized node embeddings.
        self.node_embeddings = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.xavier_uniform_(self.node_embeddings.weight)
        with torch.no_grad():
            self.node_embeddings.weight.div_(self.node_embeddings.weight.norm(p=2, dim=1, keepdim=True))

        # Encoder (GAT).
        self.encoder = GraphEncoder(input_dim=embedding_dim, 
                                    hidden_dims=[embedding_dim], 
                                    output_dim=embedding_dim, 
                                    layer_type = encoder_layer)

        # Decoder (MLP).
        self.decoder = MLPDecoder(input_dim=embedding_dim)
        
        # Save node to embedding mapping in AE object, to use it for forward pass.
        self.node_to_embeddings = node_to_embeddings_index
        self.variable_embeddings_ids = variable_embeddings_ids
        self.bin_embedding_ids = bin_embedding_ids

    def forward(self, graph_edges, pos_neg_edges):
        # Retrieve actual node embeddings as sum of corresponding variable- and value-level embeddings.
        component_1_embeddings = self.node_embeddings(self.node_to_embeddings[:, 0]) 
        component_2_embeddings = self.node_embeddings(self.node_to_embeddings[:, 1])
        node_embeds = component_1_embeddings + component_2_embeddings
        
        # Normalize node embeddings.
        # node_embeds = self.node_embeddings.weight  # [num_nodes, embedding_dim]
        node_embeds = node_embeds / node_embeds.norm(p=2, dim=1, keepdim=True)

        # Encode to latent embeddings using GAT and normalize again.
        latent_embeds = self.encoder(node_embeds, graph_edges)
        latent_embeds = latent_embeds / latent_embeds.norm(p=2, dim=1, keepdim=True)

        # Decode predicted link probabilities for positive and negative edges.
        link_preds = self.decoder(latent_embeds, pos_neg_edges)

        return link_preds

    @torch.no_grad()
    def get_embeddings(self, graph_edges, normalize=True):
        self.eval()  # set to eval mode (disable dropout/batchnorm if any)

        # Retrieve actual node embeddings as sum of corresponding variable- and value-level embeddings.
        component_1_embeddings = self.node_embeddings(self.node_to_embeddings[:, 0]) 
        component_2_embeddings = self.node_embeddings(self.node_to_embeddings[:, 1])
        node_embeds = component_1_embeddings + component_2_embeddings
        
        #node_embeds = self.node_embeddings.weight
        node_embeds = node_embeds / node_embeds.norm(p=2, dim=1, keepdim=True)

        latent_embeds = self.encoder(node_embeds, graph_edges)

        if normalize:
            latent_embeds = latent_embeds / latent_embeds.norm(p=2, dim=1, keepdim=True)

        # Additionally extract variable embeddings (does not need encoder pass).
        variable_embeddings = self.node_embeddings(self.variable_embeddings_ids)
        bin_embeddings = self.node_embeddings(self.bin_embedding_ids)

        return latent_embeds, variable_embeddings, bin_embeddings

    def get_decoder(self):
        """Returns the trained MLPDecoder module."""
        return self.decoder