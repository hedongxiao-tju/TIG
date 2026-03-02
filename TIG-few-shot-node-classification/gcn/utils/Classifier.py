import torch
import torch.nn.functional as F
import torch_scatter

class Classifier:
    def __init__(self,  ft_in, nb_classes):
        self.nb_classes = nb_classes
        self.ave = torch.FloatTensor(nb_classes, ft_in).cuda()
    def forward(self,embs, labels, train=0):
        if train == 1:
            self.ave = averageemb(labels=labels, rawret=embs, nb_class=self.nb_classes)
        rawret = torch.cat((embs, self.ave), dim=0)
        rawret = torch.cosine_similarity(rawret.unsqueeze(1), rawret.unsqueeze(0), dim=-1)
        ret = rawret[:embs.shape[0], embs.shape[0]:]
        ret = F.softmax(ret, dim=1)
        return ret


def averageemb(labels, rawret, nb_class):
    retlabel = torch_scatter.scatter(src=rawret, index=labels, dim=0, reduce='mean')
    return retlabel


