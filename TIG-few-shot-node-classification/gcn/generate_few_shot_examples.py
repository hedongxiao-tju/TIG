
import torch
import os

def create_few_data_folder(task_num, dataset_name, data, num_classes):
    for k in range(1, 6):
        k_shot_folder = './Experiment/sample_data/Node/' + dataset_name + '/' + str(k) + '_shot'
        os.makedirs(k_shot_folder, exist_ok=True)
        for i in range(1, task_num + 1):
            folder = os.path.join(k_shot_folder, str(i))
            if not os.path.exists(folder):
                os.makedirs(folder)
                node_sample_and_save(data, k, folder, num_classes)
                print(str(k) + ' shot ' + str(i) + ' th is saved!!')


def node_sample_and_save(data, k, folder, num_classes):
    labels = data.y.to('cpu')
    num_test = int(0.9 * data.num_nodes)
    if num_test < 1000:
        num_test = int(0.7 * data.num_nodes)
    indices = torch.randperm(data.num_nodes)
    test_idx = indices[:num_test]
    test_labels = labels[test_idx]
    remaining_idx = indices[num_test:]
    remaining_labels = labels[remaining_idx]
    train_idx = torch.cat([remaining_idx[remaining_labels == i][:k] for i in range(num_classes)])
    shuffled_indices = torch.randperm(train_idx.size(0))
    train_idx = train_idx[shuffled_indices]
    train_labels = labels[train_idx]
    torch.save(train_idx, os.path.join(folder, 'train_idx.pt'))
    torch.save(train_labels, os.path.join(folder, 'train_labels.pt'))
    torch.save(test_idx, os.path.join(folder, 'test_idx.pt'))
    torch.save(test_labels, os.path.join(folder, 'test_labels.pt'))

