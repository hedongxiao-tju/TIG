import torch
from GCL.eval import get_split, LREvaluator

def freeze_test(z, label, train_ratio=0.1, test_ratio=0.8, test_num=10):
    r = torch.zeros(test_num)
    for num in range(test_num):
        split = get_split(num_samples=z.size()[0], train_ratio=train_ratio, test_ratio=test_ratio)
        result = LREvaluator(num_epochs=10000)(z, label, split)
        r[num] = result['micro_f1']
    print('mean:', str(r.mean()), 'std:', str(r.std()))
    return r.mean(), r.std()


