import numpy as np
import scipy as sp
import scipy.sparse as sp
import torch
import torch.nn.functional as F


def compute_svd(adj_matrix, args):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    U, S, Vt = sp.linalg.svds(adj_matrix, k=args.svd_k, random_state=args.seed)

    U = torch.tensor(U.copy(), dtype=torch.float32)
    S = torch.tensor(S.copy(), dtype=torch.float32)
    Vt = torch.tensor(Vt.copy(), dtype=torch.float32)

    sqrt_Lambda_S = torch.sqrt(torch.diag(S))

    result = U @ sqrt_Lambda_S + Vt.T @ sqrt_Lambda_S

    return result

