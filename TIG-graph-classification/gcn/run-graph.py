import numpy as np

from eval import  freeze_test
from itertools import chain
from model import center_away_loss, neighbor_close_loss, center_to0_loss
from utils.seed import set_seed
from model import GCN, TIG_Encoder, Linear
from utils.get_train_test_data import get_data
import sys
import torch
import argparse
import torch.optim as optim
import warnings
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import torch_geometric.nn as pyg_nn
from sklearn.preprocessing import normalize


def evaluate_graph_embeddings_using_svm(embeddings, labels):
    result = []
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    for train_index, test_index in kf.split(embeddings, labels):
        x_train = embeddings[train_index]
        x_test = embeddings[test_index]
        y_train = labels[train_index]
        y_test = labels[test_index]
        params = {"C": [1e-3, 1e-2, 1e-1, 1, 10]}
        svc = SVC(random_state=42)
        clf = GridSearchCV(svc, params)
        clf.fit(x_train, y_train)

        preds = clf.predict(x_test)
        f1 = f1_score(y_test, preds, average="micro")
        result.append(f1)
    test_f1 = np.mean(result)
    test_std = np.std(result)

    return test_f1, test_std

def main():
    def train():
        model_Linear.train()
        model_TIG_Encoder.train()
        for data in train_loader:
            data = data.cuda()
            optimizer.zero_grad()
            out = model_Linear(data.x)
            z = model_TIG_Encoder(out, data.edge_index)

            assert args.loss_type in ['InfoNCE', 'Relation', 'Scattering']
            if args.loss_type == 'InfoNCE':
                loss_train = model_TIG_Encoder.loss_no_Aug(z, data.edge_index)
            elif args.loss_type == 'Relation':
                loss1 = center_to0_loss(z)
                loss2 = neighbor_close_loss(z, data.edge_index)
                loss_train = loss1 + args.loss_ratio * loss2
            else:
                loss1 = center_away_loss(z)
                loss2 = neighbor_close_loss(z, data.edge_index)
                loss_train = loss1 + args.loss_ratio * loss2


            loss_train.backward()
            optimizer.step()
            print('Epoch: {:04d}'.format(epoch + 1),'loss_train: {:.4f}'.format(loss_train.item()))


    def test():
        model_Linear.eval()
        model_TIG_Encoder.eval()
        x_list = []
        y_list = []
        with torch.no_grad():
            for i, batch_g in enumerate(test_loader):
                batch_g = batch_g.cuda()
                out = model_Linear(batch_g.x)
                z = model_TIG_Encoder(out, batch_g.edge_index)
                out = pyg_nn.global_mean_pool(z, batch_g.batch)
                y_list.append(batch_g.y.cpu().numpy())
                x_list.append(out.cpu().numpy())
        x = np.concatenate(x_list, axis=0)
        y = np.concatenate(y_list, axis=0)

        num_repeats = 50
        acc_list = []
        f1_list = []
        for _ in range(num_repeats):
            x_proto, x_test, y_proto, y_test = train_test_split(x, y, test_size=0.8, stratify=y)

            unique_classes = np.unique(y)
            prototypes = {}

            for cls in unique_classes:
                prototypes[cls] = np.mean(x_proto[y_proto == cls], axis=0)

            for cls in prototypes:
                prototypes[cls] = normalize(prototypes[cls].reshape(1, -1))[0]

            x_test_norm = normalize(x_test)
            predictions = []

            for x_vec in x_test_norm:
                similarities = {cls: np.dot(x_vec, prototypes[cls]) for cls in prototypes}
                pred_cls = max(similarities, key=similarities.get)
                predictions.append(pred_cls)

            acc = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, average='macro')

            acc_list.append(acc)
            f1_list.append(f1)


        acc_mean, acc_std = np.mean(acc_list), np.std(acc_list)
        f1_mean, f1_std = np.mean(f1_list), np.std(f1_list)

        print(f"Average Accuracy: {acc_mean:.4f}, Standard Deviation: {acc_std:.6f}")
        print(f"Average F1-score: {f1_mean:.4f}, Standard Deviation: {f1_std:.6f}")







    train_data, test_dataset = get_data(args)
    model_Linear = Linear(train_data[0].x.shape[1], args.GCN_out_dimension)
    model_TIG_Encoder = TIG_Encoder(args.GCN_out_dimension, args.TIG_Encoder_out_dimension, tau=args.tau)
    if torch.cuda.is_available():
        model_Linear.cuda()
        model_TIG_Encoder.cuda()
    optimizer = optim.Adam(chain(model_TIG_Encoder.parameters(), model_Linear.parameters()), lr=args.lr,weight_decay=args.weight_decay)

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

    for epoch in range(args.epochs):
        train()
    optimizer.zero_grad()

    for test_data in test_dataset:
        test_loader = DataLoader(test_data, batch_size=128)
        test()




    print(args.svd_k)
    print(args.GCN_out_dimension)
    print(args.TIG_Encoder_out_dimension)
    print(args.lr)
    print(args.weight_decay)
    print(args.epochs)
    print(args.loss_ratio)
    print('--------------------------------------------------------------------------------------------------------------------------------------')


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()

    parser.add_argument('--tau', type=float, default=0.5, help='tau.')
    parser.add_argument('--lr', type=float, default=0.00001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
    parser.add_argument('--seed', type=int, default=0, help='seed.')

    parser.add_argument('--svd_k', type=int, default=256, help='SVD_out_dimension')
    parser.add_argument('--GCN_out_dimension', type=int, default=1024, help='GCN_out_dimension')
    parser.add_argument('--TIG_Encoder_out_dimension', type=int, default=1024, help='TIG_Encoder_out_dimension')

    parser.add_argument('--loss_type', type=str, default='Scattering', help='loss_type.(InfoNCE)(Relation)(Scattering)')  
    parser.add_argument('--loss_ratio', type=float, default=0.2, help='Proportion of loss composition')



    parser.add_argument('--datasets_dir', type=str, default='../datasets', help='datasets dir.')
    parser.add_argument('--train_dataset', type=str, default='IMDB-BINARY', help='train dataset.')
    parser.add_argument('--datasets_name', type=str, nargs='+', default=['IMDB-BINARY','COLLAB'], help='test datasets.')
    #


    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    set_seed(args.seed)
    main();