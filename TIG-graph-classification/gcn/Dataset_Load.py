import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, TUDataset
from collections import namedtuple, Counter
import torch.nn.functional as F

def load_dataset(dataset_name, dataset_dir, args):

    print('Dataloader: Loading Dataset', dataset_name)
    assert dataset_name in ['Cora', 'CiteSeer', 'PubMed','Photo','Computers','CS','Physics', 'PROTEINS','COLLAB']
    
    if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(dataset_dir, name=dataset_name, 
                            transform=T.NormalizeFeatures())

        
    elif dataset_name in ['Photo','Computers']:
        dataset = Amazon(dataset_dir, name=dataset_name, 
                         transform=T.NormalizeFeatures())
        
    elif dataset_name in ['CS','Physics']:
        dataset = Coauthor(dataset_dir, name=dataset_name, 
                           transform=T.NormalizeFeatures())
    elif dataset_name in['PROTEINS','COLLAB']:
        dataset = TUDataset(dataset_dir, name=dataset_name).shuffle()
        

    print('Dataloader: Loading success.')
    print(dataset[0])
    return dataset



import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree, add_self_loops, remove_self_loops



def load_graph_classification_dataset(dataset_name, dataset_dir, args,deg4feat=False):

    dataset_name = dataset_name.upper()
    dataset = TUDataset(root=dataset_dir, name=dataset_name)

    graph = dataset[0]

    if graph.x is None:
        if hasattr(graph, "node_labels") and not deg4feat:
#             print("Using node labels as node features")

            feature_dim = max(g.node_labels.max().item() for g in dataset) + 1

            new_dataset = []
            for g in dataset:
                g = g.clone()
                g.x = F.one_hot(g.node_labels, num_classes=feature_dim).float()
                new_dataset.append(g)

            dataset = new_dataset
        else:
#             print("Using degree as node features")

            max_degree = 0
            for g in dataset:
                deg = degree(g.edge_index[0], num_nodes=g.num_nodes).long()
                max_degree = max(max_degree, deg.max().item())

            feature_dim = max_degree + 1

            new_dataset = []
            for g in dataset:
                g = g.clone()
                deg = degree(g.edge_index[0], num_nodes=g.num_nodes).long()
                degree_feat = F.one_hot(deg, num_classes=feature_dim).float()
                g.x = degree_feat
                new_dataset.append(g)

            dataset = new_dataset

    else:
        pass
#         print("Using existing `x` as node features")


    dataset = [g.clone() for g in dataset]
    for g in dataset:
        g.edge_index, _ = remove_self_loops(g.edge_index)
        g.edge_index, _ = add_self_loops(g.edge_index, num_nodes=g.num_nodes)

    new_dataset = []

    for g in dataset:
        g = g.clone()

        if g.x.size(1) < args.svd_k:
            padding = args.svd_k + 2 - g.x.size(1)


            g.x = F.pad(g.x, (0, padding), value=0)

        new_dataset.append(g)

    dataset = new_dataset

    return dataset

