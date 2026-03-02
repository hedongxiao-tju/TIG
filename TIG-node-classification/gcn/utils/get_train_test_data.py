from utils.make_Ax import make_Ax_individual
from Dataset_Load import load_dataset
import torch


def get_data(args):
    with torch.no_grad():
        test_dataset = []
        for dataset_name in args.datasets_name:
            test_data = load_dataset(dataset_name, args.datasets_dir,args)[0].cuda() if torch.cuda.is_available() else load_dataset(dataset_name, args.datasets_dir,args)[0]
            test_data_Ax = make_Ax_individual(test_data, args)
            test_dataset.append((test_data.detach().cpu(),test_data_Ax.detach().cpu()))

        if args.train_dataset in args.datasets_name:
            train_data = test_dataset[args.datasets_name.index(args.train_dataset)]
        else:
            train_data_original = load_dataset(args.train_dataset, args.datasets_dir,args)[0].cuda() if torch.cuda.is_available() else \
            load_dataset(args.train_dataset, args.datasets_dir,args)[0]
            train_data_Ax = make_Ax_individual(train_data_original, args)
            train_data = (train_data_original.detach().cpu(),train_data_Ax.detach().cpu())
    return train_data,test_dataset

def get_train_data(args,name):
    with torch.no_grad():
        train_data_original = load_dataset(name, args.datasets_dir,args)[0].cuda() if torch.cuda.is_available() else \
        load_dataset(name, args.datasets_dir,args)[0]
        train_data_Ax = make_Ax_individual(train_data_original, args)
        train_data = (train_data_original.detach().cpu(),train_data_Ax.detach().cpu())
    return train_data