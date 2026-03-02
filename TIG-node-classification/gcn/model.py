import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.layer = nn.ModuleList()
        self.layer.append(GCNConv(in_channels, out_channels))

    def forward(self, x, edge_index):
        x = self.layer[-1](x, edge_index)
        return x

class TIG_Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels,tau: float = 0.5, k = 2):
        super(TIG_Encoder, self).__init__()

        self.layer = nn.ModuleList()
        self.k = k
        self.layer.append(GCNConv(in_channels, out_channels))
        for _ in range(1,k):
            self.layer.append(GCNConv(out_channels, out_channels))
        self.act = nn.PReLU()
        self.tau: float = tau
        self.fc1 = torch.nn.Linear(out_channels, out_channels)
        self.fc2 = torch.nn.Linear(out_channels, out_channels)


    def forward(self, x: torch.Tensor,edge_index: torch.Tensor, batch = None, prompt =None) -> torch.Tensor:
        for i in range(self.k-1):
            x = self.act(self.layer[i](x, edge_index))
        x = self.layer[-1](x, edge_index)
        if batch is None:
            return x
        else:
            if prompt is not None:
                x = prompt(x)
            graph_emb = global_mean_pool(x, batch.long())
            return graph_emb

    def semi_loss_no_Aug(self, z: torch.Tensor, edge_index: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        adj = torch.sparse_coo_tensor(edge_index, torch.ones_like(edge_index[0, :]).to(edge_index.device)
                                      , (z.size()[0],z.size()[0])).to_dense().to(edge_index.device).float()
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        sim = f(F.normalize(z) @ F.normalize(z).T)
        pos_sim = (adj @ sim).diag()
        neg_sim = sim.sum(1)
        return -torch.log(pos_sim / neg_sim)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = self.act(self.fc1(z))
        return self.fc2(z)

    def loss_no_Aug(self, z: torch.Tensor, edge_index: torch.Tensor):
        h = self.projection(z)
        l = self.semi_loss_no_Aug(h, edge_index)
        ret = l.mean()
        return ret

def center_away_loss(z):
    z = F.normalize(z)
    z_mean = z.mean(dim=0, keepdim=True)
    distances = torch.norm(z - z_mean, dim=1)
    loss = -distances.mean()
    return loss

def center_to0_loss(z):
    z = F.normalize(z)
    z_mean = z.mean(dim=0, keepdim=True)
    distances = torch.norm(z_mean, dim=1)
    loss = distances
    return loss

def neighbor_close_loss(z,edge_index):
    z = F.normalize(z)
    src, dst = edge_index
    distances = torch.norm(z[src] - z[dst], dim=1)
    loss = distances.mean()
    return loss

