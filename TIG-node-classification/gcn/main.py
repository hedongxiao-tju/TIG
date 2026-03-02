from eval import  freeze_test
from itertools import chain
from model import center_away_loss, neighbor_close_loss, center_to0_loss
from utils.seed import set_seed
from model import GCN, TIG_Encoder
from utils.get_train_test_data import get_data
import sys
import torch
import argparse
import torch.optim as optim
import warnings


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
        for test_data in test_dataset:
            test_data[0].cuda()
            test_data[1].cuda()
            set_seed(args.seed)

            val_data_original = test_data[0]
            val_data_Ax = test_data[1]
            with torch.no_grad():
                val_data_original.x = model_GCN(val_data_Ax.x, val_data_Ax.edge_index)[:val_data_original.x.shape[0], :]
                output = model_TIG_Encoder(val_data_original.x, val_data_original.edge_index).detach()

            freeze_test(output, val_data_original.y, train_ratio=0.1, test_ratio=0.8, test_num=10)
            test_data[0].detach().cpu()
            test_data[1].detach().cpu()

    train_data, test_dataset = get_data(args)
    model_GCN = GCN(train_data[1].x.shape[1], args.GCN_out_dimension)
    model_TIG_Encoder = TIG_Encoder(args.GCN_out_dimension, args.TIG_Encoder_out_dimension, tau=args.tau)
    if torch.cuda.is_available():
        model_GCN.cuda()
        model_TIG_Encoder.cuda()
    optimizer = optim.Adam(chain(model_TIG_Encoder.parameters(), model_GCN.parameters()), lr=args.lr,weight_decay=args.weight_decay)
    train_data[0].cuda()
    train_data[1].cuda()
    for epoch in range(args.epochs):
        train()
    optimizer.zero_grad()
    train_data[0].detach().cpu()
    train_data[1].detach().cpu()
    test()
    print('--------------------------------------------------------------------------------------------------------------------------------------')


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()

    parser.add_argument('--tau', type=float, default=0.5, help='tau.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
    parser.add_argument('--seed', type=int, default=0, help='seed.')

    parser.add_argument('--svd_k', type=int, default=256, help='SVD_out_dimension')
    parser.add_argument('--GCN_out_dimension', type=int, default=512, help='GCN_out_dimension')
    parser.add_argument('--TIG_Encoder_out_dimension', type=int, default=1024, help='TIG_Encoder_out_dimension')

    parser.add_argument('--loss_type', type=str, default='Scattering', help='loss_type.(InfoNCE)(Relation)(Scattering)')
    parser.add_argument('--loss_ratio', type=float, default=0.15, help='Proportion of loss composition')



    parser.add_argument('--datasets_dir', type=str, default='datasets', help='datasets dir.')
    parser.add_argument('--train_dataset', type=str, default='CiteSeer', help='train dataset.')
    parser.add_argument('--datasets_name', type=str, nargs='+', default=['Computers'], help='test datasets.')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    set_seed(args.seed)
    main();