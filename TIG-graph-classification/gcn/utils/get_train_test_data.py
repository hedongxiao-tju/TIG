import numpy as np

from utils.make_Ax import make_Ax_individual
from Dataset_Load import load_dataset,load_graph_classification_dataset
import torch
import numpy as np
import scipy as sp
import scipy.sparse as sp


def get_data(args):

    with torch.no_grad():
        test_dataset = []
        for dataset_name in args.datasets_name:
            test_data = load_graph_classification_dataset(dataset_name, args.datasets_dir,args)
            graph_num = len(test_data)
            x_x = torch.zeros(test_data[0].x.shape[1], test_data[0].x.shape[1])
            for i in range(graph_num):
                ax = make_Ax_individual(test_data[i], args)
                ax_tensor = torch.from_numpy(ax.toarray()).float()
                x_x += ax_tensor


            row_min = x_x.min(dim=1, keepdim=True)[0]  # (10, 1)
            row_max = x_x.max(dim=1, keepdim=True)[0]  # (10, 1)


            row_range = row_max - row_min
            row_range[row_range == 0] = 1

            adj_matrix = (x_x - row_min) / row_range

            adj_matrix_np = adj_matrix.cpu().numpy()

            U, S, Vt = sp.linalg.svds(adj_matrix_np, k=args.svd_k, random_state=args.seed)


            U = torch.tensor(U.copy(), dtype=torch.float32)
            S = torch.tensor(S.copy(), dtype=torch.float32)
            Vt = torch.tensor(Vt.copy(), dtype=torch.float32)

            sqrt_Lambda_S = torch.sqrt(torch.diag(S))

            result = U @ sqrt_Lambda_S + Vt.T @ sqrt_Lambda_S
            for i in range(graph_num):
                test_data[i].x = test_data[i].x @ result

            test_dataset.append(test_data)


        if args.train_dataset in args.datasets_name:
            train_data = test_dataset[args.datasets_name.index(args.train_dataset)]
        else:
            train_data = load_graph_classification_dataset(args.train_dataset, args.datasets_dir,args)
            graph_num = len(train_data)
            x_x = torch.zeros(train_data[0].x.shape[1], train_data[0].x.shape[1])
            for i in range(graph_num):
                ax = make_Ax_individual(train_data[i], args)
                ax_tensor = torch.from_numpy(ax.toarray()).float()
                x_x += ax_tensor


            row_min = x_x.min(dim=1, keepdim=True)[0]  # (10, 1)
            row_max = x_x.max(dim=1, keepdim=True)[0]  # (10, 1)


            row_range = row_max - row_min
            row_range[row_range == 0] = 1

            adj_matrix = (x_x - row_min) / row_range

            adj_matrix_np = adj_matrix.cpu().numpy()

            U, S, Vt = sp.linalg.svds(adj_matrix_np, k=args.svd_k, random_state=args.seed)

            U = torch.tensor(U.copy(), dtype=torch.float32)
            S = torch.tensor(S.copy(), dtype=torch.float32)
            Vt = torch.tensor(Vt.copy(), dtype=torch.float32)

            sqrt_Lambda_S = torch.sqrt(torch.diag(S))

            result = U @ sqrt_Lambda_S + Vt.T @ sqrt_Lambda_S
            for i in range(graph_num):
                train_data[i].x = train_data[i].x @ result
    return train_data,test_dataset