import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Amazon, Coauthor

def load_dataset(dataset_name, dataset_dir, args):
    print('Dataloader: Loading Dataset', dataset_name)
    if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(dataset_dir, name=dataset_name,
                            transform=T.NormalizeFeatures())

    elif dataset_name in ['Photo', 'Computers']:
        dataset = Amazon(dataset_dir, name=dataset_name,
                         transform=T.NormalizeFeatures())

    elif dataset_name in ['CS', 'Physics']:
        dataset = Coauthor(dataset_dir, name=dataset_name,
                           transform=T.NormalizeFeatures())
    return dataset






