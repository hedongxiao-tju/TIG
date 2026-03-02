from GCL.eval import get_split, LREvaluator
import torch
from itertools import chain
from model import center_away_loss, neighbor_close_loss, center_to0_loss
from utils.seed import set_seed
from model import GCN, TIG_Encoder
from utils.get_train_test_data import get_data
import sys
import argparse
import torch.optim as optim
import numpy as np
import warnings
from utils.Classifier import Classifier
from generate_few_shot_examples import create_few_data_folder

def main():
    def train():
        model_GCN.train()
        model_TIG_Encoder.train()
        optimizer.zero_grad()
        train_data_original = train_data[0]
        train_data_Ax = train_data[1]
        train_data_original.x = model_GCN(train_data_Ax.x, train_data_Ax.edge_index)[:train_data_original.x.shape[0], :]
        z = model_TIG_Encoder(train_data_original.x, train_data_original.edge_index)
        assert args.loss_type in ['InfoNCE', 'Relation', 'Scattering']
        if  args.loss_type == 'InfoNCE':
            loss_train = model_TIG_Encoder.loss_no_Aug(z, train_data_original.edge_index)
        elif args.loss_type == 'Relation':
            loss1 = center_to0_loss(z)
            loss2 = neighbor_close_loss(z, train_data_original.edge_index)
            loss_train = loss1 + args.loss_ratio * loss2
        else:
            loss1 = center_away_loss(z)
            loss2 = neighbor_close_loss(z, train_data_original.edge_index)
            loss_train = loss1 + args.loss_ratio * loss2
        loss_train.backward()
        optimizer.step()
        # print('Epoch: {:04d}'.format(epoch + 1),'loss_train: {:.4f}'.format(loss_train.item()))

    def test():
        model_GCN.eval()
        model_TIG_Encoder.eval()
        test_data_index = 0
        for test_data in test_dataset:
            test_data[0].cuda()
            test_data[1].cuda()
            set_seed(args.seed)
            dataset_name = args.datasets_name[test_data_index]
            test_data_index += 1
            test_data_original = test_data[0]
            test_data_Ax = test_data[1]
            with torch.no_grad():
                test_data_original.x = model_GCN(test_data_Ax.x, test_data_Ax.edge_index)[:test_data_original.x.shape[0],:].detach()
            log = Classifier(ft_in=test_data_original.x.shape[1], nb_classes=torch.max(test_data_original.y).item() + 1)
            num_trials = 500
            # create_few_data_folder(num_trials, dataset_name, test_data_original, torch.max(test_data_original.y).item() + 1)
            shot_num = args.shot_num
            acc_list = []
            for i in range(num_trials):
                sample_data_foler_path = "./Experiment/sample_data/Node/{}/{}_shot/{}".format(dataset_name, shot_num,i + 1)
                idx_train = torch.load(f"{sample_data_foler_path}/train_idx.pt").type(torch.long).to('cuda')
                train_lbls = torch.load(f"{sample_data_foler_path}/train_labels.pt").type(torch.long).squeeze().to('cuda')
                idx_test = torch.load(f"{sample_data_foler_path}/test_idx.pt").type(torch.long).to('cuda')
                test_lbls = torch.load(f"{sample_data_foler_path}/test_labels.pt").type(torch.long).squeeze().to('cuda')
                with torch.no_grad():
                    embeddings = model_TIG_Encoder(test_data_original.x, test_data_original.edge_index).detach()
                logits = log.forward(embeddings[idx_train], test_data_original.y[idx_train], train=1).float().cuda()

                batch_size = 500
                total_correct = 0
                total_samples = 0
                for i in range(0, len(idx_test), batch_size):
                    batch_indices = idx_test[i:i + batch_size]
                    logits = log.forward(embeddings[batch_indices], test_data_original.y[batch_indices]).float().cuda()
                    preds = torch.argmax(logits, dim=1).cuda()
                    total_correct += torch.sum(preds == test_data_original.y[batch_indices]).item()
                    total_samples += len(batch_indices)
                acc = total_correct / total_samples
                acc_list.append(acc)
            average_acc = sum(acc_list) / num_trials
            print(f"Average ACC over {num_trials} trials: {average_acc:.4f}")
            std_acc = np.std(acc_list)
            print(f"Standard Deviation of ACC over {num_trials} trials: {std_acc:.4f}")
            test_data[0].detach().cpu()
            test_data[1].detach().cpu()

    train_data,test_dataset=get_data(args)
    model_GCN = GCN(train_data[1].x.shape[1], args.GCN_out_dimension)
    model_TIG_Encoder =TIG_Encoder(args.GCN_out_dimension, args.TIG_Encoder_out_dimension, tau=args.tau)
    if torch.cuda.is_available():
        model_GCN.cuda()
        model_TIG_Encoder.cuda()
    optimizer = optim.Adam(chain(model_TIG_Encoder.parameters(), model_GCN.parameters()), lr=args.lr,
                           weight_decay=args.weight_decay)
    train_data[0].cuda()
    train_data[1].cuda()
    for epoch in range(args.epochs):
        train()
    optimizer.zero_grad()
    train_data[0].detach().cpu()
    train_data[1].detach().cpu()
    test()
    print(args.train_dataset)
    print(args.datasets_name)
    print('___________________________________________________________________________________________________')




if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()

    parser.add_argument('--tau', type=float, default=0.5, help='tau.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--seed', type=int, default=0, help='seed.')

    parser.add_argument('--svd_k', type=int, default=256, help='SVD_out_dimension')
    parser.add_argument('--GCN_out_dimension', type=int, default=512, help='GCN_out_dimension')
    parser.add_argument('--TIG_Encoder_out_dimension', type=int, default=1024,
                        help='TIG_Encoder_out_dimension')

    parser.add_argument('--loss_type', type=str, default='Scattering',help='loss_type.(InfoNCE)(Relation)(Scattering)')
    parser.add_argument('--loss_ratio', type=float, default=0.1, help='Proportion of loss composition')



    parser.add_argument('--shot_num', type=int, default=3, help='shot_num')

    parser.add_argument('--datasets_dir', type=str, default='datasets', help='datasets dir.')
    parser.add_argument('--train_dataset', type=str, default='PubMed', help='train dataset.')
    parser.add_argument('--datasets_name', type=str, nargs='+', default=['Cora'],help='test datasets.')
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    set_seed(args.seed)
    main();