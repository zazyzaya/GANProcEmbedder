import torch 
from torch import nn 
from torch_geometric.nn import GCNConv

class NodeGenerator(nn.Module):
    '''
    Given input encoding of process name and noise (optional) output
    an embedding that looks like something the embedder would generate
    '''
    def __init__(self, in_feats, static_dim, hidden_feats, out_feats) -> None:
        # Blank argument so signature matches other gens
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_feats+static_dim, hidden_feats),
            nn.Dropout(0.25, inplace=True),
            nn.RReLU(),
            nn.Linear(hidden_feats, hidden_feats),
            nn.Dropout(0.25, inplace=True),
            nn.RReLU(),
            nn.Linear(hidden_feats, out_feats),
            nn.Sigmoid()
        )   

        self.static_dim = static_dim

    def forward(self, graph):
        x = graph.x 
        if self.static_dim > 0:
            x = torch.cat([
                x, torch.rand(x.size(0), self.static_dim)
            ], dim=1)
        
        return self.net(x)


class GCNDiscriminator(nn.Module):
    '''
    Given graph input, and node embeddings predict whether those 
    nodes have real looking embeddings. When trained against generator
    and evaluated on a real graph, legitimate embeddings of 
    illegitimate processes should look weird to it. Thus, it is used
    as an anomaly detector.
    '''
    def __init__(self, emb_feats, hidden_feats):
        super().__init__()

        self.gnn1 = GCNConv(emb_feats, hidden_feats)
        self.gnn2 = GCNConv(hidden_feats, hidden_feats)
        self.lin = nn.Linear(hidden_feats, 1)
        
        self.drop = nn.Dropout(0.25)

    def forward(self, z, graph):
        x = self.drop(torch.tanh(self.gnn1(z, graph.edge_index)))
        x = self.drop(torch.tanh(self.gnn2(x, graph.edge_index)))
        
        # Sigmoid applied later. Using BCE loss w Logits
        return self.lin(x)


class FFNNDiscriminator(nn.Module):
    '''
    For ablation study. Doesn't work very well, so we know 
    GNNs add value (Gets stuck around AUC 0.5, AP 0.07, ie random)
    '''
    def __init__(self, emb_feats, hidden_feats, **kwargs):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(emb_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, 1)
        )
    
    def forward(self, z, _):
        '''
        Ignore graph inputs
        '''
        return self.net(z)